from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import gradio as gr

# Load Datasets
white_wine = pd.read_csv('./resources/winequality-white.csv', sep=';')
red_wine = pd.read_csv('./resources/winequality-red.csv', sep=';')

# Combine Datasets
white_wine['type'] = '0'
red_wine['type'] = '1'
wine_data = pd.concat([white_wine, red_wine], ignore_index=True)

# Preprocessing
wine_data.drop("free sulfur dioxide", axis=1, inplace=True)

wine_data['quality'] = wine_data['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = wine_data.drop('quality', axis=1)

y = wine_data['quality']

# Train-Test Split, Scaling, SMOTE, and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
rf_classifier_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_smote.fit(X_train_res, y_train_res)
y_pred_smote = rf_classifier_smote.predict(X_test)

def predict_quality(
        fixed_acidity: float, 
        volatile_acidity: float, 
        citric_acid: float, 
        residual_sugar: float, 
        chlorides: float, 
        total_sulfur_dioxide: float,
        density: float, 
        ph: float, 
        sulphates: float,
        alcohol: float, 
        color: str) -> str:

    if color.lower() not in ['red', 'white']:
        return
    wine_color = 0 if color.lower() == 'red' else 1

    features = [
        'fixed acidity', 
        'volatile acidity', 
        'citric acid', 
        'residual sugar',
        'chlorides', 
        'total sulfur dioxide', 
        'density', 
        'pH', 
        'sulphates',
        'alcohol', 
        'type'
    ]

    input_values = [
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
            wine_color
        ]
    ]

    input_data = pd.DataFrame(input_values, columns=features)

    features_scaled = scaler.transform(input_data)
    
    prediction = rf_classifier_smote.predict(features_scaled)
    return "High Quality" if prediction[0] == 1 else "Low Quality"


iface = gr.Interface(
    fn=predict_quality,
    inputs=[
        gr.Number(label="Fixed Acidity"),
        gr.Number(label="Volatile Acidity"),
        gr.Number(label="Citric Acid"),
        gr.Number(label="Residual Sugar"),
        gr.Number(label="Chlorides"),
        gr.Number(label="Total Sulfur Dioxide"),
        gr.Number(label="Density"),
        gr.Number(label="pH"),
        gr.Number(label="Sulphates"),
        gr.Number(label="Alcohol"),
        gr.Dropdown(choices=["red", "white"], label="Color")
    ],
    outputs=gr.Label(label="Predicted Quality"),
    title="Wine Quality Predictor",
    description="Predict the quality of wine based on its chemical characteristics."
)

iface.launch(share=True)