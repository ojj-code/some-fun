import numpy as np
import pandas as pd
import joblib
import sys
from sklearn.neighbors import NearestNeighbors

# Load saved objects from training_v7.py
global_model = joblib.load('global_model.joblib')
models = joblib.load('models.joblib')
centroids_scaled = joblib.load('centroids_scaled.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
best_weight = joblib.load('best_weight.joblib')

# Initialize nearest neighbors model (k=3 as per training script)
k_neighbors = 3
nn_model = NearestNeighbors(n_neighbors=k_neighbors)
nn_model.fit(centroids_scaled)

# Function to check if receipt ends in .49 or .99
def is_special_receipt(amount):
    cents = round((amount % 1) * 100)
    return 1 if cents in [49, 99] else 0

# Preprocess a single input, matching training_v7.py
def preprocess_input(trip_duration_days, miles_traveled, total_receipts_amount):
    df = pd.DataFrame([{
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }])
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['is_five_days'] = (df['trip_duration_days'] == 5).astype(int)
    df['receipt_sweet_spot'] = ((df['total_receipts_amount'] >= 600) &
                               (df['total_receipts_amount'] <= 800)).astype(int)
    df['high_mileage'] = (df['miles_traveled'] > 500).astype(int)
    df['day_of_month'] = np.random.randint(1, 31, size=len(df))  # Random as in training
    df['temporal_factor'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
    df['special_receipt'] = df['total_receipts_amount'].apply(is_special_receipt)
    df['trip_length_cat'] = pd.cut(df['trip_duration_days'], bins=[0, 2, 5, 7, 14],
                                   labels=['very_short', 'short', 'medium', 'long'])
    df = pd.get_dummies(df, columns=['trip_length_cat'], prefix='trip_length')
    df['duration_times_miles'] = df['trip_duration_days'] * df['miles_traveled']
    df['receipts_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1e-5)
    df['miles_times_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    X_new = df.reindex(columns=feature_names, fill_value=0)
    return X_new.values

# Predict for a single input using global and local models
def predict_single(trip_duration_days, miles_traveled, total_receipts_amount):
    X_new = preprocess_input(trip_duration_days, miles_traveled, total_receipts_amount)
    X_new_scaled = scaler.transform(X_new)
    distances, indices = nn_model.kneighbors(X_new_scaled)
    selected_model_indices = indices[0]
    preds = [models[idx].predict(X_new)[0] for idx in selected_model_indices]
    weights = 1 / (distances[0] + 1e-10)
    local_pred = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
    global_pred = global_model.predict(X_new)[0]
    final_pred = best_weight * global_pred + (1 - best_weight) * local_pred
    return final_pred

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    trip_duration_days = float(sys.argv[1])
    miles_traveled = float(sys.argv[2])
    total_receipts_amount = float(sys.argv[3])
    prediction = predict_single(trip_duration_days, miles_traveled, total_receipts_amount)
    print(f"{prediction:.2f}")