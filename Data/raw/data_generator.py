import os
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ==========================
# CONFIGURATION
# ==========================

OUTPUT_FOLDER = "generated_data"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "loan_datasetforest.csv")

CHUNK_SIZE = 100000


# ==========================
# MASTER DATA
# ==========================

GENDERS = [
    "Male",
    "Female"
]

MARITAL_STATUS = [
    "Single",
    "Married",
    "Divorced",
    "Widowed"
]

EDUCATION = [
    "10th",
    "12th",
    "Diploma",
    "Bachelor",
    "Master",
    "PhD"
]

EMPLOYMENT = [
    "Government",
    "Private",
    "Self-Employed",
    "Business",
    "Freelancer",
    "Unemployed"
]

OCCUPATIONS = [
    "Software Engineer",
    "Teacher",
    "Doctor",
    "Nurse",
    "Police",
    "Farmer",
    "Business Owner",
    "Electrician",
    "Mechanic",
    "Civil Engineer",
    "Lawyer",
    "Accountant",
    "Sales Executive",
    "Data Analyst",
    "Student"
]

LOAN_PURPOSE = [
    "Home",
    "Education",
    "Vehicle",
    "Business",
    "Medical",
    "Marriage",
    "Personal"
]

HOUSE_TYPES = [
    "Owned",
    "Rented",
    "Leased"
]

CITY_TIERS = [
    "Tier-1",
    "Tier-2",
    "Tier-3"
]

STATES = [
    "Andhra Pradesh",
    "Telangana",
    "Karnataka",
    "Tamil Nadu",
    "Kerala",
    "Maharashtra",
    "Delhi",
    "Gujarat",
    "Punjab",
    "Rajasthan"
]


# ==========================
# HELPER FUNCTIONS
# ==========================

def yes_no(prob=0.5):
    return "Yes" if random.random() < prob else "No"


def random_customer_id():
    return "CUST" + str(random.randint(10000000, 99999999))


def random_pan():
    letters = ''.join(random.choices(string.ascii_uppercase, k=5))
    numbers = ''.join(random.choices(string.digits, k=4))
    last = random.choice(string.ascii_uppercase)
    return letters + numbers + last


def random_aadhaar():
    return ''.join(random.choices(string.digits, k=12))


def random_mobile():
    return "9" + ''.join(random.choices(string.digits, k=9))


def generate_credit_score(income, experience):

    score = 550

    if income > 300000:
        score += 25

    if income > 600000:
        score += 35

    if income > 1000000:
        score += 40

    if experience > 5:
        score += 20

    if experience > 10:
        score += 25

    score += random.randint(-60, 60)

    score = max(300, min(score, 900))

    return score


def calculate_interest(credit):

    if credit >= 800:
        return round(random.uniform(7.0,8.5),2)

    elif credit >=750:
        return round(random.uniform(8.0,9.5),2)

    elif credit>=700:
        return round(random.uniform(9.0,11.0),2)

    elif credit>=650:
        return round(random.uniform(11.0,13.0),2)

    else:
        return round(random.uniform(13.0,18.0),2)


def approval_probability(
        credit,
        income,
        dti,
        defaults,
        savings,
        collateral):

    score = 0

    score += credit/900*35

    score += min(income/2000000,1)*20

    score += max(0,(100-dti))/100*15

    score += min(savings/5000000,1)*10

    if collateral=="Yes":
        score +=10

    if defaults=="Yes":
        score -=30

    score += random.uniform(-5,5)

    score=max(0,min(score,100))

    return round(score,2)


def risk_category(prob):

    if prob>=80:
        return "Very Low"

    elif prob>=60:
        return "Low"

    elif prob>=40:
        return "Medium"

    elif prob>=20:
        return "High"

    return "Very High"


def ensure_output_folder():

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


print("="*60)
print("        LOAN APPROVAL DATASET GENERATOR")
print("="*60)
# ==========================
# GENERATE ONE RECORD
# ==========================

def generate_record():

    age = random.randint(21, 65)

    gender = random.choice(GENDERS)

    marital = random.choice(MARITAL_STATUS)

    education = random.choice(EDUCATION)

    employment = random.choices(
        EMPLOYMENT,
        weights=[15, 45, 15, 10, 10, 5],
        k=1
    )[0]

    occupation = random.choice(OCCUPATIONS)

    experience = max(0, age - 21)
    experience = min(experience, random.randint(0, experience))

    # ======================
    # Income
    # ======================

    if employment == "Government":
        annual_income = random.randint(400000, 1800000)

    elif employment == "Private":
        annual_income = random.randint(250000, 2200000)

    elif employment == "Business":
        annual_income = random.randint(400000, 5000000)

    elif employment == "Self-Employed":
        annual_income = random.randint(250000, 3000000)

    elif employment == "Freelancer":
        annual_income = random.randint(150000, 1800000)

    else:
        annual_income = random.randint(0, 200000)

    monthly_income = round(annual_income / 12, 2)

    other_income = random.randint(0, 250000)

    total_monthly_income = monthly_income + other_income / 12

    # ======================
    # Credit
    # ======================

    credit_score = generate_credit_score(
        annual_income,
        experience
    )

    credit_history = random.randint(
        1,
        max(1, age - 18)
    )

    # ======================
    # Loan
    # ======================

    loan_amount = random.randint(
        50000,
        min(
            max(100000, annual_income * 5),
            10000000
        )
    )

    loan_term = random.choice([
        12,
        24,
        36,
        48,
        60,
        84,
        120,
        180,
        240,
        360
    ])

    interest_rate = calculate_interest(
        credit_score
    )

    existing_loans = random.randint(0, 4)

    existing_emis = existing_loans * random.randint(
        2000,
        25000
    )

    monthly_expenses = random.randint(
        8000,
        90000
    )

    debt_to_income = round(
        (
            existing_emis +
            monthly_expenses
        ) /
        max(total_monthly_income, 1)
        * 100,
        2
    )

    # ======================
    # Assets
    # ======================

    savings = random.randint(
        0,
        annual_income * 3
    )

    investments = random.randint(
        0,
        annual_income * 2
    )

    bank_balance = random.randint(
        1000,
        max(1000, int(annual_income * random.uniform(0.05, 0.80)))
    )

    property_owned = yes_no(0.35)

    vehicle_owned = yes_no(0.55)

    house_type = random.choice(
        HOUSE_TYPES
    )

    collateral = yes_no(0.40)

    collateral_value = (
        random.randint(
            100000,
            15000000
        )
        if collateral == "Yes"
        else 0
    )

    # ======================
    # Location
    # ======================

    city_tier = random.choice(
        CITY_TIERS
    )

    state = random.choice(
        STATES
    )

    pincode = random.randint(
        100000,
        999999
    )

    # ======================
    # Business
    # ======================

    business_owner = (
        "Yes"
        if employment == "Business"
        else "No"
    )

    business_income = (
        random.randint(
            200000,
            5000000
        )
        if business_owner == "Yes"
        else 0
    )

    # ======================
    # Verification
    # ======================

    pan_verified = yes_no(0.96)

    aadhaar_verified = yes_no(0.98)

    mobile_verified = yes_no(0.99)

    email_verified = yes_no(0.95)

    employer_verified = (
        yes_no(0.90)
        if employment != "Unemployed"
        else "No"
    )

    gst_filed = (
        yes_no(0.75)
        if business_owner == "Yes"
        else "No"
    )

    income_tax = (
        yes_no(0.80)
        if annual_income > 500000
        else yes_no(0.30)
    )

    # ======================
    # Loan History
    # ======================

    previous_default = yes_no(0.08)

    late_payments = (
        random.randint(0, 10)
        if previous_default == "Yes"
        else random.randint(0, 2)
    )

    # ======================
    # Scores
    # ======================

    fraud_score = random.randint(
        0,
        100
    )

    transaction_score = random.randint(
        30,
        100
    )

    document_complete = random.randint(
        70,
        100
    )

    loan_purpose = random.choice(
        LOAN_PURPOSE
    )

    probability = approval_probability(
        credit_score,
        annual_income,
        debt_to_income,
        previous_default,
        savings,
        collateral
    )

    risk = risk_category(
        probability
    )

    approved = (
        "Approved"
        if probability >= random.uniform(45, 75)
        else "Rejected"
    )

    # ======================
    # Return Record
    # ======================

    return {

        "CustomerID": random_customer_id(),

        "Age": age,

        "Gender": gender,

        "MaritalStatus": marital,

        "Education": education,

        "EmploymentType": employment,

        "Occupation": occupation,

        "ExperienceYears": experience,

        "AnnualIncome": annual_income,

        "MonthlyIncome": round(monthly_income,2),

        "OtherIncome": other_income,

        "CreditScore": credit_score,

        "CreditHistoryYears": credit_history,

        "LoanAmount": loan_amount,

        "LoanTermMonths": loan_term,

        "InterestRate": interest_rate,

        "DebtToIncomeRatio": debt_to_income,

        "MonthlyExpenses": monthly_expenses,

        "Savings": savings,

        "Investments": investments,

        "BankBalance": bank_balance,

        "ExistingLoans": existing_loans,

        "ExistingEMIs": existing_emis,

        "PropertyOwned": property_owned,

        "VehicleOwned": vehicle_owned,

        "HouseOwnership": house_type,

        "CityTier": city_tier,

        "State": state,

        "PINCode": pincode,

        "BusinessOwner": business_owner,

        "BusinessIncome": business_income,

        "PANVerified": pan_verified,

        "AadhaarVerified": aadhaar_verified,

        "MobileVerified": mobile_verified,

        "EmailVerified": email_verified,

        "EmployerVerified": employer_verified,

        "GSTFiled": gst_filed,

        "IncomeTaxFiled": income_tax,

        "PreviousLoanDefault": previous_default,

        "LatePayments": late_payments,

        "FraudRiskScore": fraud_score,

        "TransactionScore": transaction_score,

        "DocumentCompleteness": document_complete,

        "LoanPurpose": loan_purpose,

        "CollateralAvailable": collateral,

        "CollateralValue": collateral_value,

        "ApprovalProbability": probability,

        "RiskCategory": risk,

        "LoanApproved": approved

    }
# ==========================
# GENERATE DATASET
# ==========================

def generate_dataset(total_records):

    ensure_output_folder()

    # Remove old file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    first_chunk = True

    print("\nGenerating Dataset...\n")

    with tqdm(total=total_records, unit="rows") as pbar:

        generated = 0

        while generated < total_records:

            current_chunk = min(
                CHUNK_SIZE,
                total_records - generated
            )

            rows = []

            for _ in range(current_chunk):
                rows.append(generate_record())

            df = pd.DataFrame(rows)

            df.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk
            )

            first_chunk = False

            generated += current_chunk

            pbar.update(current_chunk)

    print("\n" + "=" * 60)
    print("Dataset Generated Successfully")
    print("=" * 60)

    print(f"Output File : {OUTPUT_FILE}")

    size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"File Size   : {size:.2f} MB")
    print(f"Total Rows  : {total_records:,}")

    print("=" * 60)


# ==========================
# MAIN
# ==========================

def main():

    print()

    while True:

        try:

            total = int(
                input(
                    "Enter number of records to generate : "
                )
            )

            if total <= 0:
                print("Please enter a positive number.\n")
                continue

            break

        except ValueError:
            print("Invalid input. Enter an integer.\n")

    start = datetime.now()

    generate_dataset(total)

    end = datetime.now()

    print(f"\nStarted : {start}")
    print(f"Finished: {end}")
    print(f"Time Taken : {end - start}")


if __name__ == "__main__":
    main()