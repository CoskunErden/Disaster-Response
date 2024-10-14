# Disaster Response Pipeline

### Summary
The "Disaster Response Pipeline" project is a machine learning-based web application designed to help disaster response organizations categorize incoming messages from disaster victims and responders. The application classifies messages into various categories such as "food," "shelter," and "medical help" to assist organizations in responding effectively and efficiently. The project consists of an ETL pipeline for data processing, a machine learning pipeline for message classification, and a Flask web application to display results and visualizations.

---

### Project Components

#### 1. ETL Pipeline
- The ETL pipeline reads and processes two datasets:
  - **messages.csv**: Contains the messages sent during disaster events.
  - **categories.csv**: Contains the categories to which each message belongs.
- The pipeline cleans and transforms the data by:
  - Removing duplicates and handling missing values.
  - Splitting the categories into individual labels.
- The processed data is then stored in an SQLite database (`DisasterResponse.db`).

#### 2. ML Pipeline
- This pipeline takes the cleaned data from the database, extracts features (using TF-IDF), and trains a machine learning model to categorize messages into multiple relevant categories.
- The pipeline includes:
  - Data preprocessing (tokenization, stopword removal, and lemmatization).
  - Model training using a multi-output classifier (**RandomForestClassifier**).
  - Hyperparameter tuning using **GridSearchCV**.
  - Model evaluation using **precision**, **recall**, and **F1-score** metrics.

#### 3. Flask Web Application
- A web app allows users to input new messages and receive category predictions.
- The web app also includes data visualizations of the training data, such as the distribution of message categories.

---

### Folder Structure

```
Root Directory 
│
├── app 
│   ├── run.py                # Flask application file 
│   └── templates/ 
│       ├── master.html        # Main page of the web app 
│       └── go.html            # Result page for classification 
│
├── data 
│   ├── disaster_messages.csv  # Dataset containing messages 
│   ├── disaster_categories.csv # Dataset containing categories 
│   └── process_data.py        # Script for data cleaning and ETL pipeline 
│
├── models 
│   └── train_classifier.py    # Script for building the machine learning model 
│
├── jpnb 
│   ├── ETL Pipeline.ipynb     # Jupyter notebook for ETL pipeline exploration 
│   └── ML Pipeline.ipynb      # Jupyter notebook for model building and tuning 
│
├── DisasterResponse.db        # SQLite database containing cleaned data 
├── README.md                  # ReadMe file (you are here!) 
├── .gitignore                 # Files to ignore in Git 
└── LICENSE                    # License for the project
```

---

### Instructions for Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/CoskunErden/Disaster-Response.git
   cd Disaster-Response
   ```

2. Run the ETL pipeline:
   - Process the data and store the results in an SQLite database:
     ```bash
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db
     ```

3. Train the machine learning model:
   - Train the classifier and save the model as a pickle file:
     ```bash
     python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
     ```

4. Run the web app:
   - Launch the Flask web app:
     ```bash
     python app/run.py
     ```
   - Open a web browser and navigate to `http://localhost:3001/` to use the app.

---

### Dataset

The datasets used in this project are:
1. **Messages Dataset**: Contains real disaster messages sent during disaster events.
2. **Categories Dataset**: Contains corresponding categories for each message, labeled for multi-category classification.

---

### Requirements

You need to install the following Python packages:

- `pandas`
- `numpy`
- `sqlalchemy`
- `scikit-learn`
- `nltk`
- `Flask`
- `plotly`
- `joblib`
- `SQLAlchemy`

Install the required libraries by running:
```bash
pip install -r requirements.txt
```

---

### Model Performance

The model has been evaluated on various metrics such as **precision**, **recall**, and **F1-score** for each of the disaster response categories. The performance metrics are reported to ensure the model's effectiveness in real-world applications.

---

### Future Improvements

- Implement further hyperparameter tuning for better accuracy.
- Add more robust NLP preprocessing techniques like word embeddings.
- Enhance the UI of the Flask web app for better user experience.

---

### License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

### Acknowledgments

Special thanks to [Figure Eight](https://appen.com/) for providing the disaster response data and to **Udacity** for designing this capstone project.

