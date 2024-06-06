import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

def generate_classification_report(current_data_path, reference_data_path, report_path):
    # Read current and reference data
    current_data = pd.read_csv(current_data_path)
    reference_data = pd.read_csv(reference_data_path)

    # Specify the target and prediction columns
    target_column = 'target'  # Specify the column containing true labels
    prediction_column = 'prediction'  # Specify the column containing predicted labels

    # Initialize the report with ClassificationPreset
    report = Report(metrics=[ClassificationPreset(target=target_column, prediction=prediction_column)])

    # Run the classification analysis
    report.run(reference_data=reference_data,
               current_data=current_data)

    # Save the report as HTML
    report.save(report_path)

if __name__ == "__main__":
    # Specify paths to current and reference data, and path to save the report
    current_data_path = "../../data/validate/preprocessed_data.csv"
    reference_data_path = "../../data/validate/reference_data.csv"
    report_path = "../../reports/classification_report.html"

    # Generate and save the classification report
    generate_classification_report(current_data_path, reference_data_path, report_path)
