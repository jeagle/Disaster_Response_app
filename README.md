# Disaster Response app

Udacity project creating an app who uses ML to process text messages.

## Installations
The project is done in Jupyter Notebook (Anaconda). Pyton 3.6.

## Motivation
Udacity project

1. Create a Data pipeline to process the data - ETL - Extract, transform, load.
2. Create a Machine Learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model
3. Display the results in a Flask app.


## Files
The project contains:

__process_data.py__ : Data pipeline: loads, cleans and saves dataframe in sql database.

__train_classifier__: ML pipeline: loads in cleaned data and trains data using 
ML (RandomForestClassifier, GridSearchCV),, saves model as pickle.

__run.py__: Runs the web app.


Licensing, Authors, Acknowledgements, Authenticity etc.
I confirm that this project is done solely by myself.
