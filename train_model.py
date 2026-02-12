import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

def train_model():
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    if not os.path.exists(csv_path):
        print("File keypoint.csv tidak ditemukan. Jalankan process_dataset.py terlebih dahulu.")
        return

    # Load data
    print("Memuat data...")
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Melatih model (Random Forest dengan optimasi)...")
    # Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,      # More trees
        max_depth=20,          # Deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced', # Handle unbalanced classes
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = 'model/keypoint_classifier/keypoint_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel disimpan ke: {model_path}")

if __name__ == "__main__":
    train_model()
