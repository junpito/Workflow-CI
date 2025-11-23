"""
Test inference script for ME/CFS Depression Model
"""
import requests
import json

# Sample data for prediction (using actual column names from train_data.csv)
data = {
    "dataframe_split": {
        "columns": [
            "age", "gender", "sleep_quality_index", "brain_fog_level", "physical_pain_score",
            "stress_level", "depression_phq9_score", "fatigue_severity_scale_score",
            "pem_duration_hours", "hours_of_sleep_per_night", "pem_present",
            "work_status", "social_activity_level", "exercise_frequency",
            "meditation_or_mindfulness"
        ],
        "data": [
            # Sample 1: ME/CFS patient (Male, high fatigue, PEM present)
            [0.75, 1.0, 0.85, 0.32, 0.91, 0.78, -0.36, 0.65, -1.02, 0.55, 0, 2.0, 1.0, 4.0, 1.0],
            # Sample 2: Depression patient (Male, low sleep quality, high depression score)
            [1.60, 1.0, 0.03, 0.98, 0.35, 0.98, 1.24, 0.70, 1.30, 0.90, 1, 2.0, 1.0, 3.0, 1.0],
            # Sample 3: Healthy control (Female, good overall scores)
            [0.50, 0.0, 0.90, 0.10, 0.20, 0.30, -1.00, 0.20, -1.50, 0.80, 0, 1.0, 3.0, 5.0, 0.0]
        ]
    }
}

# Make prediction request
url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

print("Sending prediction request...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}\n")

try:
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        predictions = response.json()
        print("✅ Prediction successful!")
        print(f"Results: {predictions}")
        print(f"\nPredicted classes: {predictions['predictions']}")
        
        # Decode predictions
        class_names = {0: "Healthy", 1: "ME/CFS", 2: "Depression"}
        print("\nDecoded predictions:")
        for i, pred in enumerate(predictions['predictions']):
            print(f"  Sample {i+1}: {class_names.get(pred, 'Unknown')} (class {pred})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection error. Make sure the Docker container is running:")
    print("   docker run -p 5000:5000 junpito/me-cfs-model:latest")
except Exception as e:
    print(f"❌ Error: {str(e)}")
