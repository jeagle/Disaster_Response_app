import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """  Input: csv filepaths(messags & categories)
        Output: merged dataframe(df) with new category columns containing '1' or '0' values
    """
    # Load in messages and categories datasets 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
        
    return messages, categories

def clean_data(messages, categories):
    """ Input: Uncleaned dataframe(df) 
        Output: Cleaned dataframe(df)
    """
    # Merge datasets joining on 'id'
    df = messages.merge(categories, on='id', how='outer')
    
    # Creating columns of categories
    categories = categories['categories'].str.split(';', expand=True) # Splitting values in the categories column on ';'
    row = categories.iloc[0] # Selecting the first row of the dataframe
    category_colnames = row.apply(lambda x: x[:-2]) # Using this row no make a list of column names for the different categories
    categories.columns = category_colnames # Renaming columns in categories dataframe
    
    # Converting category values to '1' or '0'
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new columns
    df = df.drop(['categories'], axis=1) # Drop original column
    df = pd.concat([df, categories], axis=1, join='inner') # Concatenate the original df with the new categories df
    
    # Remove duplicates
    df.drop_duplicates(['id'], inplace=True)
    
    # Remove rows in "related" column containing only zero values
    df = df[df.related != 2]
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df_clean', engine, index=False)
      


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()