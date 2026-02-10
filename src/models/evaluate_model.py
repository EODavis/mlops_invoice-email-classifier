import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "invoice_emails.csv"
    model_path = project_root / "models" / "invoice_email_classifier.joblib"
    report_path = project_root / "reports" / "evaluation_report.txt"

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X = df["text"]
    y = df["label"]

    split_idx = int(len(df) * 0.75)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with open(report_path, "w") as f:
        f.write("Invoice Email Classifier Evaluation\n")
        f.write("===================================\n\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(cm))

    print("Evaluation saved to:", report_path)
    print(report)


if __name__ == "__main__":
    main()
