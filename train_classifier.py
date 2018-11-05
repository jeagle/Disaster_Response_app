import sys
import pandas as pd
import numpy as np
import pickle

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sqlalchemy import engine, create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load dataframe from database
    df = pd.read_sql('df_clean', con=engine)
    
    # Define feature(X) and target(Y) variables 
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    
    # Define category names
    category_names = df.iloc[:, 4:].columns
    
    return X, Y, category_names

def tokenize(text):
    """ Tokenization function to process text data
    """
    tokens = word_tokenize(text) # tokenize text
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    
    # Iterate through each token
    clean_tokens = []
    for tok in tokens:    
        #Lemmatize, normalize case, and remove leading/trailing whitespace
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
     
    return clean_tokens


def build_model():
    """ 
    Building a machine learning pipeline
    """
    pipeline = Pipeline([
    
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(RandomForestClassifier()))
    
    
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
    
        'moc__estimator__n_estimators': (10, 25)     
    }

    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)    
    
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """ Input: model, x and y_test data, categtory names
        Output: report, including precision, recall and f1-score for each category
    """
    y_pred = model.predict(X_test)
    report = []
    i = 0
    for column in category_names:
        report.append(classification_report(Y_test[i], y_pred[i], target_names=[column]))
        i+=1

    print(report)

def save_model(model, model_filepath):
    """ Saving model  as a pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()