import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gradio as gr
import seaborn as sns


# Import dataset
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

# metadata 
print(wine_quality.metadata) 

# variable information 
print(wine_quality.variables)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
imp = zip(X.columns, importances)
for feature, importance in imp:
    print(f"{feature}: {importance}")


wine_data = pd.concat([X, y], axis=1)

# Assuming 'wine_data' is your DataFrame containing both features and the target
# For example, it could be the combined red and white wine dataset.

plt.figure(figsize=(10, 6))
sns.lineplot(x='quality', y='alcohol', data=wine_data)

plt.title('Alcohol Content vs. Wine Quality')
plt.xlabel('Wine Quality Score')
plt.ylabel('Alcohol Content (%)')
plt.show()
#y_pred = rf.predict(X_test)
#print(f"Accuracy: {accuracy_score(y_test, y_pred)}")