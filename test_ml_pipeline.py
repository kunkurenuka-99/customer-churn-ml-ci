import json
import os
import unittest

import joblib
import pandas as pd


def high_risk_customer():
    return pd.DataFrame([{
        "SeniorCitizen": 1, "tenure": 1, "MonthlyCharges": 95.0,
        "TotalCharges": 95.0, "gender": "Female", "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }])


def low_risk_customer():
    return pd.DataFrame([{
        "SeniorCitizen": 0, "tenure": 70, "MonthlyCharges": 60.0,
        "TotalCharges": 4200.0, "gender": "Male", "Partner": "Yes",
        "Dependents": "Yes", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    }])


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("churn_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("churn_model.pkl")
        prediction = model.predict(high_risk_customer())[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_customer_churns(self):
        model = joblib.load("churn_model.pkl")
        prediction = model.predict(high_risk_customer())[0]
        self.assertEqual(int(prediction), 1)

    def test_low_risk_customer_stays(self):
        model = joblib.load("churn_model.pkl")
        prediction = model.predict(low_risk_customer())[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
