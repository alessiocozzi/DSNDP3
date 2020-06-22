import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np



def load_data(messages_filepath, categories_filepath):
    mes_df = pd.read_csv(messages_filepath)
    cat_df = pd.read_csv(categories_filepath)
    df_main = pd.merge(mes_df,cat_df,on='id')
    return df_main 


def clean_data(df):
    cat_df = df.categories.str.split(pat=';',expand=True)
    row_one = cat_df.iloc[0,:]
    cat_cols = row_one.apply(lambda x:x[:-2])
    cat_df.columns = cat_cols
    for col in cat_df:
        cat_df[col] = cat_df[col].str[-1]
        cat_df[col] = cat_df[col].astype(np.int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df,cat_df],axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    db_engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', db_engine, index=False)  


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