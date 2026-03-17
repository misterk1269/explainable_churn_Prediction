import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("Churn_Modelling.csv")

# Drop useless columns
data = data.drop(["RowNumber","CustomerId","Surname"],axis=1)

# One hot encoding
data = pd.get_dummies(data,columns=["Geography","Gender"],drop_first=True)

# Features and target
X = data.drop("Exited",axis=1)
y = data["Exited"]

feature_names = X.columns

# -----------------------------
# Train test split
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Class weights (for imbalance)
# -----------------------------
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0:weights[0],1:weights[1]}

# -----------------------------
# Neural Network Model
# -----------------------------
model = Sequential()

model.add(Input(shape=(11,)))

model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train model
# -----------------------------
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    class_weight=class_weights
)

# -----------------------------
# Prediction
# -----------------------------
y_prob = model.predict(X_test)
y_pred = (y_prob>0.5).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
print("Accuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

print("\nROC AUC Score:",roc_auc_score(y_test,y_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

# -----------------------------
# Save model
# -----------------------------
model.save("churn_model.h5")
joblib.dump(scaler,"scaler.save")

print("\nModel and scaler saved successfully")

# -----------------------------
# Explainable AI using SHAP
# -----------------------------

print("\nGenerating SHAP explanations...")

# convert scaled arrays back to dataframe for shap
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

explainer = shap.Explainer(model, X_train_df)

shap_values = explainer(X_test_df)

# Global feature importance
shap.summary_plot(shap_values, X_test_df)

# Bar feature importance
shap.summary_plot(shap_values, X_test_df, plot_type="bar")

# Individual explanation example
shap.plots.waterfall(shap_values[0])