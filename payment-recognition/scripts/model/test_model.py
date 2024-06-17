import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import onnxruntime as rt
from joblib import load

# Paths to the files
test_data_path = 'data/clean/test_data.csv'
quantized_model_path = 'models/model.quant.onnx'
joblib_model_path = 'models/model.joblib'

# Load the test data
test_df = pd.read_csv(test_data_path)

# Define label column
label_col = 'reference'

# Define features and labels
X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col]

# Load the quantized ONNX model
try:
    sess = rt.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    print(f"Input shape for ONNX model: {sess.get_inputs()[0].shape}")

    # Make predictions on the test data using the ONNX model
    y_pred = []
    batch_size = 10  # Batch size for prediction
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test.iloc[i:i+batch_size].values.astype(np.float32)
        pred = sess.run(None, {input_name: X_batch})[0]
        y_pred.extend(pred)

    # Convert predictions to integer (assuming reference is an integer)
    y_pred = np.round(y_pred).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Combine predictions with actual references for comparison
    results_df = pd.DataFrame({
        'actual_reference': y_test,
        'predicted_reference': y_pred
    })

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Sample predictions:")
    print(results_df.head(20))
except Exception as e:
    print(f"Error loading or running ONNX model: {e}")

# Load the joblib model for comparison
try:
    model = load(joblib_model_path)
    y_pred_joblib = model.predict(X_test)
    
    # Calculate accuracy for joblib model
    accuracy_joblib = accuracy_score(y_test, y_pred_joblib)
    
    # Combine predictions with actual references for comparison
    results_df_joblib = pd.DataFrame({
        'actual_reference': y_test,
        'predicted_reference': y_pred_joblib
    })
    
    # Print the results for joblib model
    print(f"Accuracy (joblib model): {accuracy_joblib:.2f}")
    print("Sample predictions (joblib model):")
    print(results_df_joblib.head(20))
except Exception as e:
    print(f"Error loading or running joblib model: {e}")
