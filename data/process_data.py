# import necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets.
    
    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.
    
    Returns:
    df: dataframe. Merged dataset of messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Clean the merged dataframe.
    
    Args:
    df: dataframe. Merged dataset of messages and categories.
    
    Returns:
    df: dataframe. Cleaned dataframe.
    """
    # Split the categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to just 0 or 1, but handle the 'related' column with a value of 2
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1].astype(int)
    
    # Replace 'related' column values of 2 with 1 (or drop the rows)
    categories['related'] = categories['related'].replace(2, 1)
    
    # Drop the original 'categories' column from df
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Assert to check if there are no duplicates
    assert len(df[df.duplicated()]) == 0, "There are duplicates in the dataframe!"
    
    return df


def save_data(df, database_filepath):
    """
    Save the cleaned data into a SQLite database.
    
    Args:
    df: dataframe. Cleaned dataframe.
    database_filepath: str. Filepath for the SQLite database.
    """
    # Create SQLite engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Save dataframe to SQLite table
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline: load data, clean data, and save data.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
cd 