# libraries
import sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


import pickle

def load_data(database_filepath):
    ''' 
    Description: load data from database using database_filepath 
    Params: filepath of the database
    Return: message column(X), categories(Y), list of category name
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df_table = pd.read_sql_table('DisasterResponse', engine)
    X_val = df_table.message.values
    rem_col = ['id', 'message', 'original', 'genre']
    y_val = df_table.loc[:, ~df_table.columns.isin(rem_col)]
    y_val.loc[:,'related'] = y_val['related'].replace(2,1)
    cat_name = y_val.columns
    return X_val, y_val, cat_name


def tokenize(text):
    ''' 
    Desc: tokenization function for processing text  
    Params: string containg untokenizated text
    Return: list of token words from text
    '''
    urls = re.findall(regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    token = word_tokenize(text)
    lem = WordNetLemmatizer()
    c_token = []
    for tok in token:
        ct = lem.lemmatize(tok).lower().strip()
        c_token.append(ct)
    return c_token


def build_model():
    '''
    Desc: model to predict category of text based on categories (36 categories available)
    Params: no parameters
    Return: model to predict classifications (36 categories available)
    '''
    pipe = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    param = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipe, param_grid=param)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Desc: Evaluate the model performance by f1 score, precision and recall
    Params: ML model X_test, Y_test, categories's name
    Return: no return 
    '''
    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=y_pred,index=Y_test.index,columns=category_names)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Desc: Save the model to a specified path
    Params: ML model, model_filepath
    Return: no return
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main function that train the classifier
    Parameters:
    arg1: the file path of the database
    arg2: the file path that the trained model will be saved
    Returns:
    None
    '''
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