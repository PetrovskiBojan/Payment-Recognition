import os
import csv
import random
from faker import Faker

# Ensure the directory exists
os.makedirs('../data/raw', exist_ok=True)

# Initialize Faker to generate fake data
fake = Faker()

# Function to generate a varied description
def generate_description(reference, iban, include_reference=True):
    patterns = [
        f"Payment for {reference}. Have a nice day",
        f"Monthly subscription {reference}",
        f"Invoice payment reference: {reference}",
        f"Payment {reference} for services",
        f"Reference number: {reference}",
        f"{reference}",
        f"Paying my dues {reference}",
        f"Subscription {reference}",
        f"Service payment {reference}",
        fake.text(max_nb_chars=50)
    ]
    
    if not include_reference:
        patterns += [
            fake.text(max_nb_chars=50),
            f"My IBAN is {iban}, thank you",
            f"Payment without reference",
            "Just a monthly payment",
            "Regular payment",
            fake.sentence()
        ]
    
    return random.choice(patterns)

# Function to generate a payment record
def generate_payment_record(base_record=None, anomalous=False):
    if base_record and not anomalous:
        # Generate a similar payment
        payor_name = base_record['name']
        iban = base_record['IBAN']
        amount = float(base_record['amount']) * random.uniform(0.95, 1.05)
        reference = base_record['reference']
        description = generate_description(reference, iban)
        day = int(base_record['day'])
        month = int(base_record['month'])
        year = int(base_record['year'])
    else:
        # Generate a new or anomalous payment
        payor_name = fake.name()
        iban = fake.iban()
        amount = round(random.uniform(10, 1000), 2)
        reference = fake.ean(length=13)
        description = generate_description(reference, iban, include_reference=not anomalous)
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(2020, 2024)

        if anomalous:
            # Introduce anomalies
            if random.choice([True, False]):
                amount *= random.uniform(1.1, 2)  # Deviate amount
            else:
                reference = fake.ean(length=8)  # Incorrect reference format
                description = generate_description(reference, iban, include_reference=False)  # Update description for the anomalous reference

    return {
        "name": payor_name,
        "IBAN": iban,
        "amount": amount,
        "reference": reference,
        "description": description,
        "day": day,
        "month": month,
        "year": year
    }

# Function to read payments from a file
def read_payments(filepath):
    payments = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                payments.append(row)
    return payments

# Generate synthetic data
def generate_data(base_records, num_normal, num_anomalous):
    data = []
    if base_records:  # Check if base_records is not empty
        for _ in range(num_normal):
            base_record = random.choice(base_records)
            data.append(generate_payment_record(base_record=base_record))
        for _ in range(num_anomalous):
            data.append(generate_payment_record(anomalous=True))
    else:
        print("Base data not found. Skipping data generation.")
    return data

# Save data to CSV file
def save_to_csv(data, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ["name", "IBAN", "amount", "reference", "description", "day", "month", "year"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data:
            writer.writerow(record)

# File path for the existing and generated data
existing_file_path = '../data/preprocessed/initial_preprocessed_data.csv'
new_file_path = '../data/raw/generated_data.csv'

# Read all payments
base_records = read_payments(existing_file_path)

# Sample 1000 records from the existing records
sampled_records = random.sample(base_records, 100) if len(base_records) >= 100 else base_records

# Generate new data
num_normal = 95
num_anomalous = 5
new_data = generate_data(sampled_records, num_normal, num_anomalous)

# Combine the old and new data
all_data = base_records + new_data

# Save the combined data to the new file
save_to_csv(all_data, new_file_path)

print(f"{len(new_data)} new records have been generated and appended. The new file contains {len(all_data)} records.")
