import json
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import sys

# Load saved objects from training (only once for efficiency)
global_model = joblib.load('global_model.joblib')
models = joblib.load('models.joblib')
centroids_scaled = joblib.load('centroids_scaled.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
best_weight = joblib.load('best_weight.joblib')

# Set up NearestNeighbors model with k=3 neighbors
k_neighbors = 3
nn_model = NearestNeighbors(n_neighbors=k_neighbors)
nn_model.fit(centroids_scaled)

# Function to check if receipt ends in .49 or .99
def is_special_receipt(amount):
    cents = round((amount % 1) * 100)
    return 1 if cents in [49, 99] else 0

# Preprocess input data into features
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
    df['day_of_month'] = np.random.randint(1, 31, size=len(df))  # Random day
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

# Predict for a single test case
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

def main():
    # Load test cases from public_cases.json
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Initialize counters and lists for metrics
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    max_error = 0.0
    max_error_case = None
    results = []
    errors = []
    
    # Process each test case
    for i, case in enumerate(test_cases):
        trip_duration_days = case['input']['trip_duration_days']
        miles_traveled = case['input']['miles_traveled']
        total_receipts_amount = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        prediction = predict_single(trip_duration_days, miles_traveled, total_receipts_amount)
        
        if prediction == "ERROR":
            errors.append(f"Case {i+1}: Prediction failed")
            continue
        
        try:
            pred_value = float(prediction)
            expected_value = float(expected)
            error = abs(pred_value - expected_value)
            
            results.append({
                'case': i+1,
                'expected': expected_value,
                'predicted': pred_value,
                'error': error
            })
            
            successful_runs += 1
            if error < 0.01:
                exact_matches += 1
            if error < 1.0:
                close_matches += 1
            total_error += error
            if error > max_error:
                max_error = error
                max_error_case = case
        except ValueError:
            errors.append(f"Case {i+1}: Invalid prediction or expected value")
    
    # Calculate average error
    if successful_runs > 0:
        avg_error = total_error / successful_runs
    else:
        avg_error = float('inf')
    
    # Calculate a simple score based on error and matches
    score = avg_error * 100 + (len(test_cases) - exact_matches) * 0.1
    
    # Print evaluation results
    print("✅ Evaluation Complete!")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Exact matches (±$0.01): {exact_matches}")
    print(f"Close matches (±$1.00): {close_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Maximum error: ${max_error:.2f}")
    print(f"Score: {score:.2f}")
    
    # Report any errors encountered
    if errors:
        print("\n⚠️ Errors encountered:")
        for error in errors[:10]:  # Show up to 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()