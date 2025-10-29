import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import gradio as gr
from ucimlrepo import fetch_ucirepo 

wine_quality = fetch_ucirepo(id=186) 

X = wine_quality.data.features 
y = wine_quality.data.targets 
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

print(wine_quality.metadata) 
print(wine_quality.variables)
print("\nQuality score distribution:")
print(y.value_counts().sort_index())

y_binary = (y >= 7).astype(int)
print(f"\nBinary classification distribution:")
print(f"Average/Poor wines (0): {(y_binary == 0).sum()}")
print(f"Good wines (1): {(y_binary == 1).sum()}")

X_engineered = X.copy()
X_engineered['alcohol_density_ratio'] = X['alcohol'] / X['density']
X_engineered['total_acidity'] = X['fixed_acidity'] + X['volatile_acidity']
X_engineered['free_to_total_sulfur'] = X['free_sulfur_dioxide'] / (X['total_sulfur_dioxide'] + 1)
X_engineered['sugar_alcohol_ratio'] = X['residual_sugar'] / X['alcohol']
X_engineered['acidity_ph_interaction'] = X['fixed_acidity'] * X['pH']

print("\nEngineered features added.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)
X_scaled = pd.DataFrame(X_scaled, columns=X_engineered.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print("\nTraining Gradient Boosting Classifier...")

gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"\nGradient Boosting Accuracy: {accuracy_gb:.4f} ({accuracy_gb*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=['Average/Poor', 'Good']))

# IMPROVEMENT 5: Also try Random Forest with better parameters
print("\nTraining optimized Random Forest...")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")

if accuracy_gb > accuracy_rf:
    best_model = gb
    best_accuracy = accuracy_gb
    model_name = "Gradient Boosting"
else:
    best_model = rf
    best_accuracy = accuracy_rf
    model_name = "Random Forest"

print(f"\n{'='*50}")
print(f"BEST MODEL: {model_name}")
print(f"ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"{'='*50}")

importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

cv_scores = cross_val_score(best_model, X_scaled, y_binary, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

cm = confusion_matrix(y_test, y_pred_gb if accuracy_gb > accuracy_rf else y_pred_rf)
print("\nConfusion Matrix:")
print(cm)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

top_features = feature_importance_df.head(10)
axes[0].barh(top_features['feature'], top_features['importance'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Top 10 Feature Importances')
axes[0].invert_yaxis()

axes[1].scatter(X['alcohol'], y, c=y_binary, cmap='RdYlGn', alpha=0.6, edgecolors='k')
axes[1].set_xlabel('Alcohol Content')
axes[1].set_ylabel('Quality Score')
axes[1].set_title('Alcohol vs Wine Quality')
axes[1].colorbar = plt.colorbar(axes[1].scatter(X['alcohol'], y, c=y_binary, cmap='RdYlGn', alpha=0.6), ax=axes[1])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Average/Poor', 'Good'],
            yticklabels=['Average/Poor', 'Good'])
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title(f'Confusion Matrix - {model_name}')

plt.tight_layout()
plt.savefig('wine_quality_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'wine_quality_analysis.png'")
plt.show()
