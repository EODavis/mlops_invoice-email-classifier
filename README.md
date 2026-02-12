# Invoice Email Classifier API (MLOps Project)

## Overview
This project is a practical **MLOps-focused NLP system** that classifies invoice and payment-related business emails into finance categories such as:

- Payment confirmed
- Payment overdue
- Invoice request
- Bank details request
- General finance inquiry

The goal is to simulate a real-world business automation pipeline where finance teams can automatically route incoming emails to the right internal unit, improving speed, efficiency, and response time.

This repository demonstrates an **end-to-end Machine Learning lifecycle**: dataset generation → model training → evaluation → deployment as an API.

---

## Business Problem
Finance and accounting teams receive large volumes of emails daily. Manually sorting these emails causes:

- delayed payment confirmations
- missed overdue reminders
- slow invoice delivery
- inefficient customer support handling

This system automates classification to enable:
- smarter ticket routing
- finance workflow automation
- faster response time
- improved payment follow-up processes

---

## Solution Summary
This project trains a text classification model using **TF-IDF + Logistic Regression**, evaluates performance using standard ML metrics, and serves predictions through a **FastAPI inference service**.

All predictions are logged for traceability.

---

## Project Features
### Machine Learning
- Synthetic invoice email dataset generator
- TF-IDF vectorisation
- Logistic Regression classification model
- Saved model artefact using `joblib`

### Evaluation
- Classification report (Precision, Recall, F1-score)
- Confusion matrix output
- Evaluation report stored as an artefact

### API Deployment
- FastAPI inference service
- Swagger documentation enabled
- `/health` endpoint for monitoring
- `/predict` endpoint for inference
- Request logging into `/logs/api_requests.log`

---

## Email Categories (Classes)
| Label | Description |
|------|-------------|
| `payment_confirmed` | Emails confirming payment was completed |
| `payment_overdue` | Emails requesting urgent payment settlement |
| `invoice_request` | Requests for invoice documents |
| `bank_details_request` | Requests for account details for payment |
| `general_finance_inquiry` | Other billing or finance questions |

---

## Repository Structure
```bash
mlops-invoice-email-classifier/
│
├── data/
│   ├── raw/
│   │   └── invoice_emails.csv
│   └── processed/
│
├── models/
│   └── invoice_email_classifier.joblib
│
├── reports/
│   └── evaluation_report.txt
│
├── logs/
│   └── api_requests.log
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   └── api/
│       └── app.py
│
├── tests/
│
├── requirements.txt
└── README.md
```
## Setup Instructions
### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/mlops-invoice-email-classifier.git
cd mlops-invoice-email-classifier
```
### 2. Create Virtual Environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
py -m pip install --upgrade pip
py -mpip install -r requirements.txt
```
## Running the Pipeline

### Step 1: Generate Dataset

Creates a synthetic dataset of invoice/payment-related emails.

```bash
python src/data/make_dataset.py
```
#### Output
```text
data/raw/invoice_emails.csv
```

