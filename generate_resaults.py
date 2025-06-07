import json
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import sys

# Load saved objects from training_v7.py (only once!)
global_model = joblib.load('global_model.joblib')
models = joblib.load('models.joblib')
centroids_scaled = joblib.load('centroids_scaled.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
best_weight = joblib.load('best_weight.joblib')

# Set up nearest neighbors model (k=3, matching training script)
k_neighbors = 3
nn_model = NearestNeighbors(n_neighbors=k_neighbors)
nn_model.fit(centroids_scaled)

# Check if receipt ends in .49 or .99
def is_special_receipt(amount):
    cents = round((amount % 1) * 100)
    return 1 if cents in [49, 99] else 0

# Preprocess input data (same as training_v7.py)
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
    df['day_of_month'] = np.random.randint(1, 31, size=len(df))  # Random, as in training
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

# Predict for one test case using global and local models
def predict_single(trip_duration_days, miles_traveled, total_receipts_amount):
    try:
        X_new = preprocess_input(trip_duration_days, miles_traveled, total_receipts_amount)
        X_new_scaled = scaler.transform(X_new)
        distances, indices = nn_model.kneighbors(X_new_scaled)
        selected_model_indices = indices[0]
        preds = [models[idx].predict(X_new)[0] for idx in selected_model_indices]
        weights = 1 / (distances[0] + 1e-10)
        local_pred = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
        global_pred = global_model.predict(X_new)[0]
        final_pred = best_weight * global_pred + (1 - best_weight) * local_pred
        return f"{final_pred:.2f}"
    except Exception as e:
        print(f"Error processing case: {e}", file=sys.stderr)
        return "ERROR"

# Main function to handle all test cases
def main():
    # Load all test cases from private_cases.json
    with open('private_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Process each case and store results
    results = []
    for i, case in enumerate(test_cases):
        trip_duration_days = case['trip_duration_days']
        miles_traveled = case['miles_traveled']
        total_receipts_amount = case['total_receipts_amount']
        prediction = predict_single(trip_duration_days, miles_traveled, total_receipts_amount)
        results.append(prediction)
        if (i + 1) % 1 == 0:  # Progress update
            print(f"Processed {i + 1}/{len(test_cases)} cases...", file=sys.stderr)
    
    # Write results to private_results.txt
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    # Final status messages
    print(f"\n✅ Results generated successfully!", file=sys.stderr)
    print(f"📄 Output saved to private_results.txt", file=sys.stderr)
    print(f"📊 Processed {len(test_cases)} test cases", file=sys.stderr)

if __name__ == "__main__":
    main()