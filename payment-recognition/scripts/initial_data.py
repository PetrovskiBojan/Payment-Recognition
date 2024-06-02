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
def generate_payment_record(anomalous=False):
    payor_name = fake.name()
    iban = fake.iban()
    amount = round(random.uniform(10, 1000), 2)
    reference = fake.ean(length=13)
    description = generate_description(reference, iban)
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

# Generate synthetic data
def generate_data(num_records):
    data = []
    for _ in range(num_records):
        if random.random() < 0.1:  # 10% anomalous data
            data.append(generate_payment_record(anomalous=True))
        else:
            data.append(generate_payment_record())
    return data

# Save data to CSV file
def save_to_csv(data, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ["name", "IBAN", "amount", "reference", "description", "day", "month", "year"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for record in data:
            writer.writerow(record)

# Generate and save data
num_records = 5000  # Number of records to generate
data = generate_data(num_records)
save_to_csv(data, '../data/raw/initial_data.csv')

print(f"{num_records} records have been generated and saved to data/raw/initial_data.csv")
