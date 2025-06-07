import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time
import random
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Load data
def load_data(file_path, has_output=True):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if has_output:
        df = pd.DataFrame([{
            'trip_duration_days': item['input']['trip_duration_days'],
            'miles_traveled': item['input']['miles_traveled'],
            'total_receipts_amount': item['input']['total_receipts_amount'],
            'expected_output': item['expected_output']
        } for item in data])
    else:
        df = pd.DataFrame([{
            'trip_duration_days': item['trip_duration_days'],
            'miles_traveled': item['miles_traveled'],
            'total_receipts_amount': item['total_receipts_amount']
        } for item in data])
    return df

# Check if receipt ends in .49 or .99
def is_special_receipt(amount):
    cents = round((amount % 1) * 100)
    return 1 if cents in [49, 99] else 0

# Preprocess data with enhanced features
def preprocess_data(df):
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['is_five_days'] = (df['trip_duration_days'] == 5).astype(int)
    df['receipt_sweet_spot'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['high_mileage'] = (df['miles_traveled'] > 500).astype(int)
    df['day_of_month'] = np.random.randint(1, 31, size=len(df))
    df['temporal_factor'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
    df['special_receipt'] = df['total_receipts_amount'].apply(is_special_receipt)
    df['trip_length_cat'] = pd.cut(df['trip_duration_days'], bins=[0, 2, 5, 7, 14], 
                                   labels=['very_short', 'short', 'medium', 'long'])
    df = pd.get_dummies(df, columns=['trip_length_cat'], prefix='trip_length')
    df['duration_times_miles'] = df['trip_duration_days'] * df['miles_traveled']
    df['receipts_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1e-5)
    df['miles_times_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    return df

# Evaluation function
def evaluate(predictions, expected):
    predictions = np.array(predictions)
    expected = np.array(expected)
    exact_matches = np.sum(np.abs(predictions - expected) < 0.01)
    total_error = np.sum(np.abs(predictions - expected))
    avg_error = total_error / len(expected)
    score = avg_error * 100 + (len(expected) - exact_matches) * 0.1
    return {'exact_matches': exact_matches, 'avg_error': avg_error, 'score': score}

# Train a small decision tree
def train_small_model(X_subset, y_subset):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_subset, y_subset)
    return model

# Main training function
def build_and_evaluate_model(file_path='public_cases.json'):
    df = load_data(file_path, has_output=True)
    df = preprocess_data(df)
    X = df.drop('expected_output', axis=1)
    y = df['expected_output']
    feature_names = X.columns
    train_size = int(0.8 * len(df))
    train_indices = random.sample(range(len(df)), train_size)
    val_indices = [i for i in range(len(df)) if i not in train_indices]
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train global model
    global_model = RandomForestRegressor(n_estimators=100, random_state=42)
    global_model.fit(X_train, y_train)
    
    # Train local models
    models = []
    subsets = []
    start_time = time.time()
    subset_size = 10
    k_neighbors = 3
    
    while time.time() - start_time < 6 * 600:
        duration_bins = pd.cut(X_train['trip_duration_days'], bins=5)
        indices = []
        for _, group in X_train.groupby(duration_bins, observed=False):
            if len(group) >= 2:
                indices.extend(random.sample(group.index.tolist(), min(2, len(group))))
        if len(indices) < subset_size:
            available_indices = list(set(X_train.index) - set(indices))
            additional_indices = random.sample(available_indices, subset_size - len(indices))
            indices.extend(additional_indices)
        
        X_subset = X_train.loc[indices]
        y_subset = y_train.loc[indices]
        model = train_small_model(X_subset, y_subset)
        models.append(model)
        subsets.append(X_subset)
    
    # Compute centroids
    subset_scaled = [scaler.transform(subset) for subset in subsets]
    centroids_scaled = np.array([np.mean(subset, axis=0) for subset in subset_scaled])
    
    # Save models and objects
    joblib.dump(global_model, 'global_model.joblib')
    joblib.dump(models, 'models.joblib')
    joblib.dump(centroids_scaled, 'centroids_scaled.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    
    # Optimize weight on validation set
    nn_model = NearestNeighbors(n_neighbors=k_neighbors)
    nn_model.fit(centroids_scaled)
    best_score = float('inf')
    best_weight = 0
    for w in np.linspace(0.1, 0.9, 9):
        predictions = []
        for i in range(len(X_val)):
            X_new = X_val.iloc[i].values.reshape(1, -1)
            X_new_scaled = X_val_scaled[i].reshape(1, -1)
            distances, indices = nn_model.kneighbors(X_new_scaled)
            selected_model_indices = indices[0]
            preds = [models[idx].predict(X_new)[0] for idx in selected_model_indices]
            weights = 1 / (distances[0] + 1e-10)
            local_pred = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
            global_pred = global_model.predict(X_new)[0]
            final_pred = w * global_pred + (1 - w) * local_pred
            predictions.append(final_pred)
        metrics = evaluate(predictions, y_val)
        if metrics['score'] < best_score:
            best_score = metrics['score']
            best_weight = w
    
    joblib.dump(best_weight, 'best_weight.joblib')
    
    # Final validation with best weight
    predictions = []
    for i in range(len(X_val)):
        X_new = X_val.iloc[i].values.reshape(1, -1)
        X_new_scaled = X_val_scaled[i].reshape(1, -1)
        distances, indices = nn_model.kneighbors(X_new_scaled)
        selected_model_indices = indices[0]
        preds = [models[idx].predict(X_new)[0] for idx in selected_model_indices]
        weights = 1 / (distances[0] + 1e-10)
        local_pred = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
        global_pred = global_model.predict(X_new)[0]
        final_pred = best_weight * global_pred + (1 - best_weight) * local_pred
        predictions.append(final_pred)
    
    metrics = evaluate(predictions, y_val)
    
    print(f"\n### Results After 10 Minutes:")
    print(f"Number of local models: {len(models)}")
    print(f"Exact Matches: {metrics['exact_matches']}")
    print(f"Average Error: ${metrics['avg_error']:.2f}")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Best global model weight: {best_weight:.2f}")
    print("Models and objects saved for prediction.")

if __name__ == "__main__":
    build_and_evaluate_model()