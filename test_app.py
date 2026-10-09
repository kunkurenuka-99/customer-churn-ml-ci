import unittest

from app import app


HIGH_RISK = {
    "SeniorCitizen": 1, "tenure": 1, "MonthlyCharges": 95.0,
    "TotalCharges": 95.0, "gender": "Female", "Partner": "No",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

LOW_RISK = {
    "SeniorCitizen": 0, "tenure": 70, "MonthlyCharges": 60.0,
    "TotalCharges": 4200.0, "gender": "Male", "Partner": "Yes",
    "Dependents": "Yes", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_risk_prediction(self):
        response = self.client.post("/predict", json=HIGH_RISK)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NO CHURN")

    def test_low_risk_prediction(self):
        response = self.client.post("/predict", json=LOW_RISK)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NO CHURN")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"tenure": 5, "MonthlyCharges": 50.0}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
