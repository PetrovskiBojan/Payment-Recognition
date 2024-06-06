import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from dotenv import load_dotenv
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load environment variables from .env file
load_dotenv()

# Setup MLflow to track to DagsHub
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('Payment-recognition')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['MLFLOW_TRACKING_PASSWORD']

# Load the data
train_data_path = '../../data/clean/train_data.csv'
test_data_path = '../../data/clean/test_data.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

label_col = 'reference'

# Define features and labels
X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]
X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col]

# Combine and split data for validation
X_train, X_validate, y_train, y_validate = train_test_split(
    pd.concat([X_train, X_test]), 
    pd.concat([y_train, y_test]), 
    test_size=0.2, 
    random_state=42
)

# Standardize the feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validate_scaled = scaler.transform(X_validate)

# Best hyperparameters
best_params = {
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Initialize and train the DecisionTreeClassifier with the best parameters
best_dt = DecisionTreeClassifier(**best_params)
best_dt.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = best_dt.predict(X_validate_scaled)

# Evaluate the model
accuracy = accuracy_score(y_validate, y_pred)
f1 = f1_score(y_validate, y_pred, average='weighted')
print(f"Validation set accuracy: {accuracy:.4f}")
print(f"Validation set F1 score: {f1:.4f}")

# Set the date for logging
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Define the model save path without the model filename
model_save_path = '../../models'
os.makedirs(model_save_path, exist_ok=True)

# Save the model locally
model_filename = os.path.join(model_save_path, 'dt.joblib')
joblib.dump(best_dt, model_filename)

# Log the model and metrics to MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_param("date", date)
    mlflow.sklearn.log_model(best_dt, "model"+str(date))

    # Convert the model to ONNX format and save it
    X_train_scaled_2d = X_train_scaled.reshape(-1, X_train_scaled.shape[1])
    initial_types = [('float_input', FloatTensorType([None, X_train_scaled_2d.shape[1]]))]
    onnx_model = convert_sklearn(best_dt, "decision tree model", initial_types=initial_types)
    onnx_model_path = os.path.join(model_save_path, "model.onnx")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Quantize the ONNX model
    quantized_model_path = os.path.join(model_save_path, "model.quant.onnx")
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

    # Log the ONNX models as artifacts in MLflow
    mlflow.log_artifact(onnx_model_path, "onnx_models")
    mlflow.log_artifact(quantized_model_path, "quantized_models")

    is_best_model = True

    if is_best_model:
        mlflow.set_tag("model", "PRODUCTION")

print(f"Model and ONNX models have been saved to {model_save_path}.")
print(f"Quantized ONNX model has been saved to {quantized_model_path}.")