import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from dotenv import load_dotenv
import joblib
from tqdm import tqdm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load environment variables from.env file
load_dotenv()

# Setup MLflow to track to DagsHub
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('Payment-recognition')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['MLFLOW_TRACKING_PASSWORD']

# Load the data
train_data_path = 'data/clean/train_data.csv'
test_data_path = 'data/clean/test_data.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

label_col = 'reference'

# Define features and labels
X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]
X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col]

# Combine and split data for validation
X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Define parameter grid for Decision Tree
param_grid_dt = {
    'max_depth': [3, 5, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize classifier
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
}

# Train and evaluate each classifier with progress tracking
results = {}
for name, clf in tqdm(classifiers.items(), desc="Training classifiers"):
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    print(f"Finished training {name}. Predicting...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'report': report
    }
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("="*60)
    
    # Define the model save path without the model filename
    model_save_path = '../../models'
    os.makedirs(model_save_path, exist_ok=True)

    # Save the model locally
    model_filename = os.path.join(model_save_path, 'model.joblib')
    joblib.dump(clf, model_filename)

    # Set the date for logging
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log the model and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_params(clf.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("date", date)
        mlflow.sklearn.log_model(clf, "model" + str(date))

        # Convert the model to ONNX format with a specific opset version
        # initial_types = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        # onnx_model = convert_sklearn(clf, initial_types=initial_types, target_opset=12)  # Adjusted target_opset
        # onnx_model_path = os.path.join(model_save_path, "model.onnx")
        # with open(onnx_model_path, "wb") as f:
        #     f.write(onnx_model.SerializeToString())

        # # Quantize the ONNX model
        # quantized_model_path = os.path.join(model_save_path, "model.quant.onnx")
        # quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

        # # Log the ONNX models as artifacts in MLflow
        # mlflow.log_artifact(onnx_model_path, "onnx_models")
        # mlflow.log_artifact(quantized_model_path, "quantized_models")

        # is_best_model = True

        # if is_best_model:
        #     mlflow.set_tag("model", "PRODUCTION")

print(f"Model and ONNX models have been saved to {model_save_path}.")
#print(f"Quantized ONNX model has been saved to {quantized_model_path}.")
