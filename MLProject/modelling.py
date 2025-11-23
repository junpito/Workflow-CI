"""
Model Training for CI/CD Workflow
Author: Junpito Salim

MLflow Project entry point with argparse support.
"""

import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Load preprocessed data."""
    print("Loading data...")
    train_data = pd.read_csv('me_cfs_vs_depression_preprocessing/train_data.csv')
    test_data = pd.read_csv('me_cfs_vs_depression_preprocessing/test_data.csv')
    
    X_train = train_data.drop('diagnosis', axis=1)
    y_train = train_data['diagnosis']
    X_test = test_data.drop('diagnosis', axis=1)
    y_test = test_data['diagnosis']
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test


def train_model(n_estimators, max_depth, random_state):
    """Train model with specified parameters."""
    print("="*60)
    print("CI/CD Workflow - Model Training")
    print("="*60 + "\n")
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # MLflow Project already creates experiment, just enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Workflow"):
        print(f"Training with: n_estimators={n_estimators}, max_depth={max_depth}\n")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nRun ID: {run_id}")
        
        # Log custom metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1)
        
        # Log model for Docker build
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ME_CFS_Depression_Model"
        )
    
    print("\n" + "="*60)
    print("Training Completed")
    print("="*60)
    
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    train_model(args.n_estimators, args.max_depth, args.random_state)
