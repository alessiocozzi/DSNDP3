# Disaster Response Pipeline Project

### Description

In this course, I've learned and built on data engineering skills to expand opportunities and potential as a data scientist. In this project, I have apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Project Components
There are three components in this project.

#### 1. ETL Pipeline

#### 2. ML Pipeline

#### 3. Flask Web App

### Screenshots:

![P1](/photo/p1.png)
![P2](/photo/p2.png)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
