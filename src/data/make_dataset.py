import pandas as pd
import random
from pathlib import Path

random.seed(42)

LABELS = {
    "payment_confirmed": [
        "We have paid the invoice, kindly confirm receipt",
        "Payment has been made successfully, please acknowledge",
        "Attached is proof of payment for the invoice",
        "We have completed payment for invoice number 2040",
        "Kindly confirm you received the payment",
        "The invoice has been settled, see payment slip attached",
    ],
    "payment_overdue": [
        "Your payment is overdue, please settle immediately",
        "This is a reminder that your invoice is past due",
        "We have not received payment, kindly pay as soon as possible",
        "Invoice remains unpaid despite previous reminders",
        "Please note your outstanding balance is overdue",
        "Your account is overdue, settle the invoice urgently",
    ],
    "invoice_request": [
        "Please send the invoice for last month",
        "Kindly resend the invoice document",
        "I did not receive the invoice, please share it",
        "Can you provide the invoice for this transaction?",
        "Please generate and send an invoice for my payment",
        "I need the invoice copy for accounting purposes",
    ],
    "bank_details_request": [
        "Please share your bank account details",
        "Kindly send your account number and bank name",
        "Where should we transfer the payment? Send bank details",
        "Please provide your IBAN or bank information",
        "Send your payment account details for transfer",
        "Kindly confirm your bank details for settlement",
    ],
    "general_finance_inquiry": [
        "Can you confirm the total outstanding balance?",
        "Please clarify the charges on the invoice",
        "What is the due date for this invoice?",
        "Kindly explain the breakdown of charges",
        "Can you share your finance department contact?",
        "We have a question regarding your pricing and billing",
    ]
}

NOISE = [
    "urgent",
    "asap",
    "kindly respond",
    "thanks",
    "please assist",
    "this is important",
    "waiting for your feedback",
    "please treat as priority"
]

def generate_text(label: str) -> str:
    base = random.choice(LABELS[label])
    if random.random() < 0.677:
        base += " " + random.choice(NOISE)
    if random.random() < 0.323:
        base += " " + random.choice(NOISE)
    return base.lower()

def create_dataset(n_samples=1350):
    labels = list(LABELS.keys())
    rows = []

    for _ in range(n_samples):
        label = random.choice(labels)
        text = generate_text(label)
        rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "raw" / "invoice_emails.csv"

    df = create_dataset(n_samples=1700)
    df.to_csv(output_path, index=False)

    print("Dataset created successfully!")
    print(f"Saved to: {output_path}")
    print(df.head(10))
    print(df.info)
    print(df.describe())

if __name__ == "__main__":
    main()
