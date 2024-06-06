import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_data_drift_report(current_data_path, reference_data_path, report_path):
    # Read current and reference data
    current_data = pd.read_csv(current_data_path)
    reference_data = pd.read_csv(reference_data_path)
    
    # Initialize column mapping
    column_mapping = ColumnMapping() 

    # Initialize the report with DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    
    # Run the data drift analysis
    report.run(reference_data=reference_data, 
               current_data=current_data, 
               column_mapping=column_mapping)
    
    # Save the report as HTML
    report.save_html(report_path)

if __name__ == "__main__":
    # Specify paths to current and reference data, and path to save the report
    current_data_path = "../../data/validate/preprocessed_data.csv"
    reference_data_path = "../../data/validate/reference_data.csv"
    report_path = "../../reports/data_drift_report.html"

    # Generate and save the data drift report
    generate_data_drift_report(current_data_path, reference_data_path, report_path)
