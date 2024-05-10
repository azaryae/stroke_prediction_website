from flask import Flask, render_template, request, flash, redirect, url_for
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from flask import render_template
import json

app = Flask(__name__)

# Set the secret key for session management and secure cookies
app.secret_key = 'your_secret_key_here'

# Define the upload folder
UPLOAD_FOLDER = 'data_upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the folder to save the trained model
MODEL_FOLDER = 'trained_models'
app.config['MODEL_FOLDER'] = MODEL_FOLDER

def load_dataset(filename):
    # Load the dataset from the uploaded file
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(dataset_path)
    return df

def train_model(X_train, y_train):
    # Define numerical and categorical features
    numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer to apply different transformations to numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Append Gaussian Naive Bayes classifier to preprocessing pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', GaussianNB())])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model


def save_model(model):
    # Save the trained model
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')
    joblib.dump(model, model_path)

def load_model():
    # Load the saved model
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')
    model = joblib.load(model_path)
    return model

@app.route('/')
def home():
    return render_template('pages/index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # Set the filename to 'dataset.csv'
        filename = 'dataset.csv'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File uploaded successfully')
        
        return redirect(url_for('upload_success'))
    
    return render_template('pages/upload.html')


@app.route('/upload_success')
def upload_success():
    return render_template('pages/upload_success.html')



import json

@app.route('/create_model', methods=['POST'])
def create_model():
    # Set the filename to 'dataset.csv'
    filename = 'dataset.csv'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load dataset
    df = pd.read_csv(file_path)

    # Split dataset into training and testing sets
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model using the training set
    model = train_model(X_train, y_train)

    # Save the model
    save_model(model)

    # Convert X_test and y_test to serializable format
    X_test_json = X_test.to_dict(orient='records')
    y_test_json = y_test.tolist()

    # Save X_test and y_test to JSON file
    with open('X_test.json', 'w') as file:
        json.dump(X_test_json, file)
    with open('y_test.json', 'w') as file:
        json.dump(y_test_json, file)

    flash('Model created successfully')

    # Redirect to the upload success page
    return redirect(url_for('confusion_matriks'))



import json

@app.route('/confusion_matrix')
def confusion_matriks():
    # Load the saved model
    model = load_model()

    # Load test dataset from JSON file
    with open('X_test.json', 'r') as file:
        X_test_json = json.load(file)
    with open('y_test.json', 'r') as file:
        y_test_json = json.load(file)

    # Convert X_test and y_test to DataFrame and Series
    X_test = pd.DataFrame(X_test_json)
    y_test = pd.Series(y_test_json)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Render template with confusion matrix and metrics
    return render_template('pages/confusion_matrix.html', confusion_matrix=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1)











@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load the saved model
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')
        model = joblib.load(model_path)
        
        # Get new instance data from the form
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']
        
        # Create new instance data
        new_instance_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })
        
        # Make prediction
        prediction = model.predict(new_instance_data)
        
        # Render template with prediction result
        return render_template('pages/prediction_result.html', prediction=prediction)

    return render_template('pages/predict.html')

if __name__ == '__main__':
    app.run(debug=True)
