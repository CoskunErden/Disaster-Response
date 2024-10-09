import numpy as np
import pandas as pd
import sys
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from utils import tokenize  # Assuming the tokenize function is available in utils.py
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load the data from the SQLite database, clean it, and return the features and labels.

    Args:
    database_filepath: str. The file path of the SQLite database.

    Returns:
    X (DataFrame): The features (messages).
    y_filtered (DataFrame): The filtered labels (categories with more than one class).
    """
    # Create a connection to the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the data from the 'disaster_messages' table
    df = pd.read_sql('SELECT * FROM disaster_messages', engine)
    
    # Assuming 'message' is the feature column and labels start at column 4
    X = df['message']
    y = df.iloc[:, 4:]
    
    # Identify columns with only one class and remove them
    columns_to_drop = [col for col in y.columns if len(set(y[col])) == 1]
    y = y.drop(columns=columns_to_drop, axis=1)

    return X, y

def build_model():
    """
    Build a machine learning pipeline with TfidfVectorizer and Logistic Regression,
    and set up a GridSearchCV for hyperparameter tuning.

    Returns:
    model (GridSearchCV): The model pipeline with GridSearchCV.
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    # Define hyperparameters for GridSearchCV
    parameters = {
        'vect__max_df': [0.9, 1.0],
        'clf__estimator__C': [1, 10]
    }

    # Grid search for hyperparameter tuning
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)

    return model_pipeline

def train(X, y, model):
    """
    Train the model on the training data, evaluate it on the test set,
    and output classification reports for each category.

    Args:
    X (DataFrame): The features.
    y (DataFrame): The labels.
    model (GridSearchCV): The model pipeline.

    Returns:
    model: The trained model.
    """
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Output classification report for each category
    for i, col in enumerate(Y_train.columns):
        print(f'Category: {col}')
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
    
    return model

def export_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model: The trained model.
    model_filepath: str. The file path where the model should be saved.
    """
    joblib.dump(model, model_filepath)

def run_pipeline(database_filepath, model_filepath):
    """
    Execute the full pipeline: load data, build the model, train it, and export the model.

    Args:
    database_filepath (str): The file path of the SQLite database.
    model_filepath (str): The file path where the model should be saved.
    """
    X, y = load_data(database_filepath)
    model = build_model()
    model = train(X, y, model)
    export_model(model, model_filepath)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_filepath = sys.argv[1]
        model_filepath = sys.argv[2]
        
        print(f'Running pipeline for dataset: {database_filepath}')
        run_pipeline(database_filepath, model_filepath)
        print(f'Model saved to: {model_filepath}')
    else:
        print('Please provide the filepath of the SQLite database and the filepath to save the model as arguments.')
        print('Example: python train_classifier.py data/DisasterResponse.db models/classifier.pkl')
