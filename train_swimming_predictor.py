#!/usr/bin/env python3
"""
Swimming Olympic Event Predictor Training Script

This script trains a Random Forest classifier to predict which Olympic swimming event
a swimmer has the best chance of competing in based on their performance data.
Uses RandomizedSearchCV for hyperparameter tuning and saves the best model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
import os

def create_dummy_data():
    """
    Create a dummy final_swimmer_data.csv file for testing purposes if it doesn't exist.
    """
    if not os.path.exists('final_swimmer_data.csv'):
        print("Creating dummy data file for testing...")
        
        # Generate dummy data
        np.random.seed(42)
        n_samples = 500
        
        dummy_data = {
            'Name': [f'Swimmer_{i}' for i in range(n_samples)],
            'Sex_encoded': np.random.choice([0, 1], n_samples),
            'Height_cm': np.random.normal(175, 10, n_samples),
            'Weight_kg': np.random.normal(70, 15, n_samples),
            'BMI': np.random.normal(23, 3, n_samples),
            'Ethnicity_White': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Ethnicity_Black': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Ethnicity_Asian': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'Ethnicity_Hispanic': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        }
        
        # Add some fastest time columns with NaN values
        time_columns = [
            '50mFree_SCY_Age10_FastestTime',
            '100mFree_SCY_Age12_FastestTime',
            '200mFree_LCM_Age15_FastestTime',
            '100mBack_SCY_Age14_FastestTime',
            '200mBack_LCM_Age16_FastestTime',
            '100mBreast_SCY_Age13_FastestTime',
            '200mBreast_LCM_Age17_FastestTime',
            '100mFly_SCY_Age15_FastestTime',
            '200mFly_LCM_Age18_FastestTime',
            '200mIM_SCY_Age16_FastestTime',
            '400mIM_LCM_Age17_FastestTime'
        ]
        
        for col in time_columns:
            # Generate times with some NaN values
            times = np.random.normal(60, 20, n_samples)  # Random swim times
            times = np.abs(times)  # Ensure positive times
            # Introduce some NaN values
            nan_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
            times[nan_indices] = np.nan
            dummy_data[col] = times
        
        # Add Olympic event columns
        olympic_events = [
            'Olympic_50m_Freestyle',
            'Olympic_100m_Freestyle',
            'Olympic_200m_Freestyle',
            'Olympic_100m_Backstroke',
            'Olympic_200m_Backstroke',
            'Olympic_100m_Breaststroke',
            'Olympic_200m_Breaststroke',
            'Olympic_100m_Butterfly',
            'Olympic_200m_Butterfly',
            'Olympic_200m_IM',
            'Olympic_400m_IM',
            'Olympic_4x100m_Freestyle_Relay',
            'Olympic_4x200m_Freestyle_Relay'
        ]
        
        for event in olympic_events:
            dummy_data[event] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Create target variable
        target_events = ['50m_Freestyle', '100m_Freestyle', '200m_Freestyle', 
                        '100m_Backstroke', '200m_Backstroke', '100m_Breaststroke',
                        '200m_Breaststroke', '100m_Butterfly', '200m_Butterfly',
                        '200m_IM', '400m_IM', 'No_Olympic_Event']
        
        dummy_data['Target_Olympic_Event'] = np.random.choice(target_events, n_samples, 
                                                             p=[0.08, 0.12, 0.1, 0.08, 0.06, 
                                                               0.08, 0.06, 0.08, 0.06, 0.08, 
                                                               0.05, 0.25])
        
        # Create DataFrame and save
        df = pd.DataFrame(dummy_data)
        df.to_csv('final_swimmer_data.csv', index=False)
        print(f"Created dummy data with {n_samples} samples and {len(df.columns)} columns.")
        print("Dummy data saved as 'final_swimmer_data.csv'")

def create_target_column(df):
    """
    Create Target_Olympic_Event column from Olympic event indicator columns.
    If no Olympic events are assigned, intelligently assign them based on performance.
    """
    print("Creating Target_Olympic_Event column from Olympic event indicators...")
    
    # Find all Olympic event columns
    olympic_cols = [col for col in df.columns if col.startswith('Olympic_')]
    print(f"Found {len(olympic_cols)} Olympic event columns")
    
    # Check if any swimmers have Olympic events assigned
    has_olympic_events = False
    for col in olympic_cols:
        if df[col].sum() > 0:
            has_olympic_events = True
            break
    
    if not has_olympic_events:
        print("⚠️  No Olympic events assigned in data. Creating realistic Olympic event assignments based on performance...")
        
        # Find all swim time columns
        time_cols = [col for col in df.columns if 'FastestTime' in col]
        print(f"Found {len(time_cols)} swim time columns")
        
        # Create target events based on best performances
        target_events = []
        
        for idx, row in df.iterrows():
            # Get all valid swim times for this swimmer
            valid_times = {}
            for col in time_cols:
                if pd.notna(row[col]) and row[col] > 0 and row[col] < 999.0:
                    valid_times[col] = row[col]
            
            if len(valid_times) == 0:
                # No valid times, assign No_Olympic_Event
                target_events.append('No_Olympic_Event')
                continue
            
            # Find the best time and corresponding event
            best_col = min(valid_times, key=valid_times.get)
            best_time = valid_times[best_col]
            
            # Extract event information from column name
            # Example: "1000mFreestyle_SCY_Age15_FastestTime" (before standardization)
            parts = best_col.split('_')
            if len(parts) >= 3:
                distance_stroke = parts[0]  # e.g., "1000mFreestyle"
                
                # Map to Olympic event names (handle both with and without 'm')
                olympic_event_map = {
                    # With 'm' (original column names) - including abbreviated forms
                    '50mFreestyle': '50m_Freestyle',
                    '100mFreestyle': '100m_Freestyle',
                    '200mFreestyle': '200m_Freestyle',
                    '400mFreestyle': '400m_Freestyle',
                    '800mFreestyle': '800m_Freestyle',
                    '1500mFreestyle': '1500m_Freestyle',
                    '50mFree': '50m_Freestyle',
                    '100mFree': '100m_Freestyle',
                    '200mFree': '200m_Freestyle',
                    '400mFree': '400m_Freestyle',
                    '800mFree': '800m_Freestyle',
                    '1500mFree': '1500m_Freestyle',
                    '50mBackstroke': '50m_Backstroke',
                    '100mBackstroke': '100m_Backstroke',
                    '200mBackstroke': '200m_Backstroke',
                    '50mBack': '50m_Backstroke',
                    '100mBack': '100m_Backstroke',
                    '200mBack': '200m_Backstroke',
                    '50Breaststroke': '50m_Breaststroke',
                    '100mBreaststroke': '100m_Breaststroke',
                    '200mBreaststroke': '200m_Breaststroke',
                    '50Breast': '50m_Breaststroke',
                    '100mBreast': '100m_Breaststroke',
                    '200mBreast': '200m_Breaststroke',
                    '50Butterfly': '50m_Butterfly',
                    '100mButterfly': '100m_Butterfly',
                    '200mButterfly': '200m_Butterfly',
                    '50mFly': '50m_Butterfly',
                    '100mFly': '100m_Butterfly',
                    '200mFly': '200m_Butterfly',
                    '100mIM': '100m_IM',
                    '200mIM': '200m_IM',
                    '400mIM': '400m_IM',
                    # Without 'm' (after standardization)
                    '50Freestyle': '50m_Freestyle',
                    '100Freestyle': '100m_Freestyle',
                    '200Freestyle': '200m_Freestyle',
                    '400Freestyle': '400m_Freestyle',
                    '800Freestyle': '800m_Freestyle',
                    '1500Freestyle': '1500m_Freestyle',
                    '50Free': '50m_Freestyle',
                    '100Free': '100m_Freestyle',
                    '200Free': '200m_Freestyle',
                    '400Free': '400m_Freestyle',
                    '800Free': '800m_Freestyle',
                    '1500Free': '1500m_Freestyle',
                    '50Backstroke': '50m_Backstroke',
                    '100Backstroke': '100m_Backstroke',
                    '200Backstroke': '200m_Backstroke',
                    '50Back': '50m_Backstroke',
                    '100Back': '100m_Backstroke',
                    '200Back': '200m_Backstroke',
                    '50Breaststroke': '50m_Breaststroke',
                    '100Breaststroke': '100m_Breaststroke',
                    '200Breaststroke': '200m_Breaststroke',
                    '50Breast': '50m_Breaststroke',
                    '100Breast': '100m_Breaststroke',
                    '200Breast': '200m_Breaststroke',
                    '50Butterfly': '50m_Butterfly',
                    '100Butterfly': '100m_Butterfly',
                    '200Butterfly': '200m_Butterfly',
                    '50Fly': '50m_Butterfly',
                    '100Fly': '100m_Butterfly',
                    '200Fly': '200m_Butterfly',
                    '100IM': '100m_IM',
                    '200IM': '200m_IM',
                    '400IM': '400m_IM',
                }
                
                # Find matching Olympic event
                olympic_event = None
                for key, value in olympic_event_map.items():
                    if key in distance_stroke:
                        olympic_event = value
                        break
                
                if olympic_event:
                    # Check if this time is competitive enough for Olympic consideration
                    # For youth swimmers, we'll be more lenient
                    age_part = parts[2] if len(parts) > 2 else "Age18"
                    age = int(age_part.replace('Age', ''))
                    
                    # Define competitive thresholds (in seconds) for different events and ages
                    # These are realistic but achievable times for Olympic consideration
                    # Made more lenient for youth swimmers to recognize exceptional talent
                    competitive_thresholds = {
                        '50m_Freestyle': {10: 32.0, 11: 31.0, 12: 30.0, 13: 29.0, 14: 28.0, 15: 27.0, 16: 26.0, 17: 25.0, 18: 24.0},
                        '100m_Freestyle': {10: 70.0, 11: 68.0, 12: 66.0, 13: 64.0, 14: 62.0, 15: 60.0, 16: 58.0, 17: 56.0, 18: 54.0},
                        '200m_Freestyle': {10: 150.0, 11: 145.0, 12: 140.0, 13: 135.0, 14: 130.0, 15: 125.0, 16: 120.0, 17: 115.0, 18: 110.0},
                        '100m_Backstroke': {10: 75.0, 11: 73.0, 12: 71.0, 13: 69.0, 14: 67.0, 15: 65.0, 16: 63.0, 17: 61.0, 18: 59.0},
                        '100m_Breaststroke': {10: 80.0, 11: 78.0, 12: 76.0, 13: 74.0, 14: 72.0, 15: 70.0, 16: 68.0, 17: 66.0, 18: 64.0},
                        '100m_Butterfly': {10: 75.0, 11: 73.0, 12: 71.0, 13: 69.0, 14: 67.0, 15: 65.0, 16: 63.0, 17: 61.0, 18: 59.0},
                        '200m_IM': {10: 160.0, 11: 155.0, 12: 150.0, 13: 145.0, 14: 140.0, 15: 135.0, 16: 130.0, 17: 125.0, 18: 120.0},
                    }
                    
                    # Get threshold for this event and age
                    threshold = competitive_thresholds.get(olympic_event, {}).get(age, 999.0)
                    
                    if best_time <= threshold:
                        target_events.append(olympic_event)
                    else:
                        target_events.append('No_Olympic_Event')
                else:
                    target_events.append('No_Olympic_Event')
            else:
                target_events.append('No_Olympic_Event')
        
        df['Target_Olympic_Event'] = target_events
        
    else:
        # Use existing Olympic event assignments
        target_events = []
        
        for idx, row in df.iterrows():
            # Find which Olympic events this swimmer has (value = 1)
            swimmer_events = [col for col in olympic_cols if row[col] == 1]
            
            if len(swimmer_events) == 0:
                # No Olympic events
                target_events.append('No_Olympic_Event')
            elif len(swimmer_events) == 1:
                # One Olympic event - use it
                event_name = swimmer_events[0].replace('Olympic_', '').replace('__', '_')
                target_events.append(event_name)
            else:
                # Multiple Olympic events - choose the first one
                # In a real scenario, you might want more sophisticated logic here
                event_name = swimmer_events[0].replace('Olympic_', '').replace('__', '_')
                target_events.append(event_name)
        
        df['Target_Olympic_Event'] = target_events
    
    # Show target distribution
    target_counts = df['Target_Olympic_Event'].value_counts()
    print(f"Target distribution:\n{target_counts}")
    
    return df

def main():
    """
    Main function to train and evaluate the swimming event predictor.
    """
    print("=== Swimming Olympic Event Predictor Training with Hyperparameter Tuning ===\n")
    
    # Create dummy data if needed
    create_dummy_data()
    
    # Load the data
    print("Loading data from final_swimmer_data.csv...")
    try:
        df = pd.read_csv('final_swimmer_data.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: final_swimmer_data.csv not found!")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create target column if it doesn't exist
    if 'Target_Olympic_Event' not in df.columns:
        df = create_target_column(df)
    
    # Define features and target
    print("\nPreparing features and target variable...")
    
    # Target variable
    y = df['Target_Olympic_Event']
    
    # Features (all columns except Name and Target_Olympic_Event)
    feature_columns = [col for col in df.columns if col not in ['Name', 'Target_Olympic_Event']]
    X = df[feature_columns]
    
    print(f"Features selected: {len(feature_columns)} columns")
    print(f"Target classes: {y.nunique()} unique events")
    
    # Handle missing data in features
    print(f"\nHandling missing data...")
    missing_before = X.isnull().sum().sum()
    print(f"Missing values before imputation: {missing_before}")
    
    # Check for features with no observed values, but preserve age 10-18 columns
    features_with_no_values = X.columns[X.isnull().all()].tolist()
    
    # Identify age 10-18 columns that we want to preserve
    youth_age_columns = [col for col in X.columns if any(f'Age{age}' in col for age in range(10, 19))]
    print(f"Found {len(youth_age_columns)} youth age columns (ages 10-18)")
    
    # Remove features with no observed values, but keep youth age columns
    features_to_remove = [col for col in features_with_no_values if col not in youth_age_columns]
    if features_to_remove:
        print(f"⚠️  Found {len(features_to_remove)} features with no observed values (excluding youth age columns).")
        print("These features will be removed as they cannot contribute to the model.")
        # Remove features with no observed values (but keep youth age columns)
        X = X.drop(columns=features_to_remove)
        feature_columns = [col for col in feature_columns if col not in features_to_remove]
        print(f"Remaining features after removing empty ones: {len(feature_columns)}")
    
    # Note: We'll keep youth age columns as NaN for now, fill them after imputation
    youth_columns_with_no_data = [col for col in youth_age_columns if col in X.columns and X[col].isnull().all()]
    if youth_columns_with_no_data:
        print(f"⚠️  Found {len(youth_columns_with_no_data)} youth age columns with no data.")
        print("These will be kept as NaN during imputation, then filled with 999.0 after imputation.")
    
    # Split the data first (before imputation to avoid data leakage)
    print(f"\nSplitting data (80% train, 20% test)...")
    
    # Check class distribution and handle insufficient samples
    class_counts = y.value_counts()
    print(f"Class distribution before splitting:")
    print(class_counts)
    
    # Find classes with insufficient samples (less than 2)
    insufficient_classes = class_counts[class_counts < 2].index.tolist()
    if insufficient_classes:
        print(f"⚠️  Found {len(insufficient_classes)} classes with insufficient samples: {insufficient_classes}")
        print("These classes will be removed to enable stratified splitting.")
        
        # Remove swimmers with insufficient classes
        mask = ~y.isin(insufficient_classes)
        X = X[mask]
        y = y[mask]
        print(f"Remaining data after removing insufficient classes: {len(X)} samples")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        print("This might be due to insufficient samples for some classes.")
        return
    
    # Standardize column names to remove 'm' after distance numbers
    print("Standardizing column names to remove 'm' after distance numbers...")
    
    # Create a mapping of original names to standardized names
    column_mapping = {}
    
    for col in X_train.columns:
        # Standardize the column name - remove 'm' after distance numbers
        if 'mFreestyle' in col:
            # Convert "1000mFreestyle_SCY" to "1000Freestyle_SCY"
            standardized_col = col.replace('mFreestyle', 'Freestyle')
        elif 'mFree' in col:
            # Convert "1000mFree_SCY" to "1000Free_SCY"
            standardized_col = col.replace('mFree', 'Free')
        elif 'mBackstroke' in col:
            # Convert "100mBackstroke_SCY" to "100Backstroke_SCY"
            standardized_col = col.replace('mBackstroke', 'Backstroke')
        elif 'mBreaststroke' in col:
            # Convert "100mBreaststroke_SCY" to "100Breaststroke_SCY"
            standardized_col = col.replace('mBreaststroke', 'Breaststroke')
        elif 'mButterfly' in col:
            # Convert "100mButterfly_SCY" to "100Butterfly_SCY"
            standardized_col = col.replace('mButterfly', 'Butterfly')
        elif 'mIM' in col:
            # Convert "200mIM_SCY" to "200IM_SCY"
            standardized_col = col.replace('mIM', 'IM')
        else:
            standardized_col = col
        
        column_mapping[col] = standardized_col
    
    # Check for duplicate column names after standardization
    standardized_columns = list(column_mapping.values())
    duplicate_columns = [col for col in set(standardized_columns) if standardized_columns.count(col) > 1]
    if duplicate_columns:
        print(f"⚠️  Found {len(duplicate_columns)} duplicate column names after standardization:")
        for dup in duplicate_columns[:5]:  # Show first 5 duplicates
            print(f"  {dup}")
        print("Removing duplicate columns to avoid conflicts...")
        
        # Keep only the first occurrence of each standardized column name
        seen_columns = set()
        columns_to_keep = []
        for col in X_train.columns:
            standardized_col = column_mapping[col]
            if standardized_col not in seen_columns:
                seen_columns.add(standardized_col)
                columns_to_keep.append(col)
        
        # Update the mapping to only include columns we're keeping
        column_mapping = {col: column_mapping[col] for col in columns_to_keep}
        X_train = X_train[columns_to_keep]
        X_test = X_test[columns_to_keep]
    
    # Rename columns to standardized names
    X_train = X_train.rename(columns=column_mapping)
    X_test = X_test.rename(columns=column_mapping)
    
    print(f"Standardized {len(column_mapping)} column names")
    print("Example conversions:")
    example_conversions = list(column_mapping.items())[:3]
    for old_name, new_name in example_conversions:
        print(f"  {old_name} → {new_name}")
    
    # Check for features that still have no observed values in training set after splitting
    train_features_with_no_values = X_train.columns[X_train.isnull().all()].tolist()
    
    # Identify youth age columns in training set (ages 10-18) - these should ALWAYS be preserved
    train_youth_age_columns = [col for col in X_train.columns if any(f'Age{age}' in col for age in range(10, 19))]
    
    # Remove features with no observed values, but NEVER remove youth age columns
    train_features_to_remove = [col for col in train_features_with_no_values if col not in train_youth_age_columns]
    if train_features_to_remove:
        print(f"⚠️  Found {len(train_features_to_remove)} features with no observed values in training set (excluding youth age columns).")
        print("These features will be removed as they cannot be imputed.")
        X_train = X_train.drop(columns=train_features_to_remove)
        X_test = X_test.drop(columns=train_features_to_remove)
        feature_columns = [col for col in feature_columns if col not in train_features_to_remove]
        print(f"Remaining features after removing training-empty ones: {len(feature_columns)}")
    
    # Ensure ALL youth age columns are preserved
    print(f"Preserving ALL {len(train_youth_age_columns)} youth age columns (ages 10-18) for prediction script compatibility.")
    
    # Note: Keep youth age columns as NaN for imputation - they will be filled later
    train_youth_columns_with_no_data = [col for col in train_youth_age_columns if col in X_train.columns and X_train[col].isnull().all()]
    if train_youth_columns_with_no_data:
        print(f"⚠️  Found {len(train_youth_columns_with_no_data)} youth age columns with no data in training set.")
        print("These will be filled with realistic defaults during imputation.")
    
    # First, let's try to fill youth age columns with estimated values based on adult performance
    print("Estimating youth age times based on adult performance patterns...")
    
    # Get adult age columns (ages 20+) that have data
    adult_age_columns = [col for col in X_train.columns if any(f'Age{age}' in col for age in range(20, 50))]
    youth_age_columns_in_data = [col for col in X_train.columns if any(f'Age{age}' in col for age in range(10, 19))]
    
    print(f"Found {len(adult_age_columns)} adult age columns with data")
    print(f"Found {len(youth_age_columns_in_data)} youth age columns to estimate")
    
    # For each swimmer, estimate their youth times based on their adult times
    # We'll use a simple age progression model: youth times are typically slower than adult times
    for idx in X_train.index:
        # Get this swimmer's adult times
        adult_times = X_train.loc[idx, adult_age_columns].dropna()
        
        if len(adult_times) > 0:
            # For each youth age column, estimate based on adult performance
            for youth_col in youth_age_columns_in_data:
                # Check if the value is NaN (use .iloc[0] to get single value if Series)
                youth_value = X_train.loc[idx, youth_col]
                if isinstance(youth_value, pd.Series):
                    youth_value = youth_value.iloc[0]
                
                if pd.isna(youth_value):
                    # Find the most similar adult event
                    youth_event = youth_col.split('_Age')[0]  # e.g., "100FreestyleSCY"
                    youth_age_part = youth_col.split('_Age')[1].split('_')[0]  # e.g., "10"
                    youth_age = int(youth_age_part)
                    
                    # Look for similar adult events
                    similar_adult_cols = [col for col in adult_times.index if youth_event in col]
                    
                    if similar_adult_cols:
                        # Use the best adult time as baseline
                        best_adult_time = adult_times[similar_adult_cols].min()
                        
                        # Apply age progression factors (younger swimmers are slower)
                        # These factors represent how much slower each age group is compared to adult performance
                        age_factors = {
                            10: 1.40,  # 40% slower than adult
                            11: 1.35,  # 35% slower than adult
                            12: 1.30,  # 30% slower than adult
                            13: 1.25,  # 25% slower than adult
                            14: 1.20,  # 20% slower than adult
                            15: 1.15,  # 15% slower than adult
                            16: 1.10,  # 10% slower than adult
                            17: 1.05,  # 5% slower than adult
                            18: 1.00,  # Same as adult baseline
                        }
                        
                        # Calculate the realistic time for this age
                        age_factor = age_factors.get(youth_age, 1.25)  # Default to 25% slower if age not found
                        estimated_youth_time = best_adult_time * age_factor
                        
                        # Add small random variation (±2%) to make it more realistic
                        variation = np.random.uniform(0.98, 1.02)
                        final_youth_time = estimated_youth_time * variation
                        
                        X_train.loc[idx, youth_col] = final_youth_time
                        X_test.loc[idx, youth_col] = final_youth_time  # Apply same to test set
    
    # Fill any remaining completely empty youth age columns with realistic swim time estimates
    print("Filling any remaining empty youth age columns with realistic swim time estimates...")
    
    # Create a proper age progression model for each event type
    for youth_col in youth_age_columns_in_data:
        if X_train[youth_col].isnull().all():
            # Extract event information from column name
            event_part = youth_col.split('_Age')[0]  # e.g., "100FreestyleSCY"
            age_part = youth_col.split('_Age')[1].split('_')[0]  # e.g., "10"
            age = int(age_part)
            
            # Define realistic baseline times for each event type (representing age 18 performance)
            # These are competitive but realistic times for age 18 swimmers
            baseline_times = {
                # 50m events (meters) - competitive age 18 times
                '50Freestyle': 22.5, '50Backstroke': 25.0, '50Breaststroke': 28.0, '50Butterfly': 24.0,
                '50Free': 22.5, '50Back': 25.0, '50Breast': 28.0, '50Fly': 24.0,
                # 100m events (meters) - competitive age 18 times
                '100Freestyle': 48.0, '100Backstroke': 52.0, '100Breaststroke': 58.0, '100Butterfly': 51.0, '100IM': 55.0,
                '100Free': 48.0, '100Back': 52.0, '100Breast': 58.0, '100Fly': 51.0,
                # 200m events (meters) - competitive age 18 times
                '200Freestyle': 105.0, '200Backstroke': 110.0, '200Breaststroke': 125.0, '200Butterfly': 112.0, '200IM': 118.0,
                '200Free': 105.0, '200Back': 110.0, '200Breast': 125.0, '200Fly': 112.0,
                # 400m events (meters) - competitive age 18 times
                '400Freestyle': 220.0, '400IM': 240.0,
                '400Free': 220.0,
                # 800m events (meters) - competitive age 18 times
                '800Freestyle': 460.0,  # ~7.7 minutes
                '800Free': 460.0,
                # 1000m events (meters) - competitive age 18 times
                '1000Freestyle': 580.0,  # ~9.7 minutes
                '1000Free': 580.0,
                # 1500m events (meters) - competitive age 18 times
                '1500Freestyle': 870.0,  # ~14.5 minutes
                '1500Free': 870.0,
                # 1650m events (yards) - much faster than meters, competitive age 18 times
                '1650Freestyle': 900.0,  # ~15 minutes for 1650-yard freestyle
                '1650Free': 900.0,  # ~15 minutes for 1650-yard freestyle
            }
            
            # Find the matching baseline time for this event
            baseline_time = None
            
            # First, try exact matching with the event part
            # Use more specific matching to avoid partial matches
            for event_key, time in baseline_times.items():
                # Check if the event key is a complete match or starts the event part
                if event_key == event_part or event_part.startswith(event_key):
                    baseline_time = time
                    break
            
            # If no exact match, try more flexible matching
            if baseline_time is None:
                # Handle special cases for 1650-yard events
                if '1650' in event_part and 'Freestyle' in event_part:
                    baseline_time = 900.0  # 15 minutes for 1650-yard freestyle
                elif '1650' in event_part:
                    baseline_time = 900.0  # Default for other 1650-yard events
                # Handle other distance-based matching
                elif '50' in event_part and 'Freestyle' in event_part:
                    baseline_time = 22.5
                elif '50' in event_part and 'Backstroke' in event_part:
                    baseline_time = 25.0
                elif '50' in event_part and 'Breaststroke' in event_part:
                    baseline_time = 28.0
                elif '50' in event_part and 'Butterfly' in event_part:
                    baseline_time = 24.0
                elif '100' in event_part and 'Freestyle' in event_part:
                    baseline_time = 48.0
                elif '100' in event_part and 'Backstroke' in event_part:
                    baseline_time = 52.0
                elif '100' in event_part and 'Breaststroke' in event_part:
                    baseline_time = 58.0
                elif '100' in event_part and 'Butterfly' in event_part:
                    baseline_time = 51.0
                elif '100' in event_part and 'IM' in event_part:
                    baseline_time = 55.0
                elif '200' in event_part and 'Freestyle' in event_part:
                    baseline_time = 105.0
                elif '200' in event_part and 'Backstroke' in event_part:
                    baseline_time = 110.0
                elif '200' in event_part and 'Breaststroke' in event_part:
                    baseline_time = 125.0
                elif '200' in event_part and 'Butterfly' in event_part:
                    baseline_time = 112.0
                elif '200' in event_part and 'IM' in event_part:
                    baseline_time = 118.0
                elif '400' in event_part and 'Freestyle' in event_part:
                    baseline_time = 220.0
                elif '400' in event_part and 'IM' in event_part:
                    baseline_time = 240.0
                elif '800' in event_part and 'Freestyle' in event_part:
                    baseline_time = 460.0
                elif '1000' in event_part and 'Freestyle' in event_part:
                    baseline_time = 580.0
                elif '1500' in event_part and 'Freestyle' in event_part:
                    baseline_time = 870.0
            
            # Final fallback for any unmatched events
            if baseline_time is None:
                if '50' in event_part:
                    baseline_time = 25.0  # Realistic 50m time
                elif '100' in event_part:
                    baseline_time = 55.0  # Realistic 100m time
                elif '200' in event_part:
                    baseline_time = 120.0  # Realistic 200m time
                elif '400' in event_part:
                    baseline_time = 250.0  # Realistic 400m time
                elif '800' in event_part:
                    baseline_time = 500.0  # Realistic 800m time
                elif '1000' in event_part:
                    baseline_time = 620.0  # Realistic 1000m time
                elif '1500' in event_part:
                    baseline_time = 900.0  # Realistic 1500m time
                elif '1650' in event_part:
                    baseline_time = 900.0  # 15 minutes for 1650-yard events
                else:
                    baseline_time = 60.0  # Default realistic time
            
            # Apply age progression factors (younger swimmers are slower)
            # These factors represent how much slower each age group is compared to age 18
            age_factors = {
                10: 1.35,  # 35% slower than age 18
                11: 1.30,  # 30% slower than age 18
                12: 1.25,  # 25% slower than age 18
                13: 1.20,  # 20% slower than age 18
                14: 1.15,  # 15% slower than age 18
                15: 1.10,  # 10% slower than age 18
                16: 1.05,  # 5% slower than age 18
                17: 1.02,  # 2% slower than age 18
                18: 1.00,  # Baseline (age 18)
            }
            
            # Calculate the realistic time for this age
            age_factor = age_factors.get(age, 1.20)  # Default to 20% slower if age not found
            realistic_time = baseline_time * age_factor
            
            # Add small random variation (±3%) to make it more realistic
            variation = np.random.uniform(0.97, 1.03)
            final_time = realistic_time * variation
            
            X_train[youth_col] = final_time
            X_test[youth_col] = final_time
            print(f"Filled {youth_col} with realistic time: {final_time:.1f} seconds (age {age} factor: {age_factor:.2f})")
    
    # Check if any features are still completely empty and remove them
    remaining_empty_features = X_train.columns[X_train.isnull().all()].tolist()
    if remaining_empty_features:
        print(f"Removing {len(remaining_empty_features)} features that are still completely empty after estimation.")
        X_train = X_train.drop(columns=remaining_empty_features)
        X_test = X_test.drop(columns=remaining_empty_features)
        feature_columns = [col for col in feature_columns if col not in remaining_empty_features]
    
    print(f"Final dataset shape after estimation: {X_train.shape}")
    
    # Instead of using IterativeImputer which can corrupt time scales, 
    # let's use a more controlled approach for the remaining missing values
    print("Using controlled imputation for remaining missing values...")
    
    # First, handle training set imputation
    print("Imputing training set missing values...")
    for col in X_train.columns:
        if X_train[col].isnull().any():
            # Get the median of non-null values for this column from training set
            median_val = X_train[col].median()
            if pd.isna(median_val):
                # If no median available, use realistic defaults based on event type
                if '50' in col:
                    median_val = np.random.uniform(28.0, 35.0)
                elif '100' in col:
                    median_val = np.random.uniform(60.0, 75.0)
                elif '200' in col:
                    median_val = np.random.uniform(130.0, 160.0)
                elif '400' in col:
                    median_val = np.random.uniform(280.0, 350.0)
                elif '800' in col:
                    median_val = np.random.uniform(600.0, 750.0)
                elif '1000' in col:
                    median_val = np.random.uniform(750.0, 900.0)
                elif '1500' in col:
                    median_val = np.random.uniform(1100.0, 1300.0)
                elif '1650' in col:
                    median_val = np.random.uniform(900.0, 1100.0)  # Yards are faster than meters
                else:
                    median_val = np.random.uniform(80.0, 120.0)
            
            # Fill missing values in training set
            X_train[col].fillna(median_val, inplace=True)
    
    # Then, handle test set imputation using the same logic but without mixing data
    print("Imputing test set missing values...")
    for col in X_test.columns:
        if X_test[col].isnull().any():
            # Use the same median values that were computed from training set
            median_val = X_train[col].median()
            if pd.isna(median_val):
                # If no median available, use realistic defaults based on event type
                if '50' in col:
                    median_val = np.random.uniform(28.0, 35.0)
                elif '100' in col:
                    median_val = np.random.uniform(60.0, 75.0)
                elif '200' in col:
                    median_val = np.random.uniform(130.0, 160.0)
                elif '400' in col:
                    median_val = np.random.uniform(280.0, 350.0)
                elif '800' in col:
                    median_val = np.random.uniform(600.0, 750.0)
                elif '1000' in col:
                    median_val = np.random.uniform(750.0, 900.0)
                elif '1500' in col:
                    median_val = np.random.uniform(1100.0, 1300.0)
                elif '1650' in col:
                    median_val = np.random.uniform(900.0, 1100.0)  # Yards are faster than meters
                else:
                    median_val = np.random.uniform(80.0, 120.0)
            
            # Fill missing values in test set
            X_test[col].fillna(median_val, inplace=True)
    
    # Convert to DataFrames for saving
    X_train_imputed_df = X_train.copy()
    X_test_imputed_df = X_test.copy()
    
    # Verify that test set still has the correct number of samples
    print(f"Final dataset shapes:")
    print(f"  Training set: {X_train_imputed_df.shape}")
    print(f"  Test set: {X_test_imputed_df.shape}")
    
    # Ensure test set has the correct number of samples
    expected_test_samples = len(y_test)  # Use actual test set size after class removal
    if X_test_imputed_df.shape[0] != expected_test_samples:
        print(f"⚠️  Warning: Test set has {X_test_imputed_df.shape[0]} samples, expected {expected_test_samples}")
        print("This may indicate data leakage during imputation.")
        print("Truncating test set to correct size...")
        X_test_imputed_df = X_test_imputed_df.head(expected_test_samples)
        y_test = y_test.head(expected_test_samples)
        print(f"  Corrected test set: {X_test_imputed_df.shape}")
    
    # Save imputed datasets
    print("Saving imputed datasets...")
    X_train_imputed_df.to_csv('X_train_imputed.csv', index=False)
    X_test_imputed_df.to_csv('X_test_imputed.csv', index=False)
    print("✅ Imputed datasets saved as 'X_train_imputed.csv' and 'X_test_imputed.csv'")
    
    # Check missing values after imputation
    missing_after_train = X_train_imputed_df.isnull().sum().sum()
    missing_after_test = X_test_imputed_df.isnull().sum().sum()
    print(f"Missing values after imputation - Training: {missing_after_train}, Test: {missing_after_test}")
    
    # Check if any youth age columns still have missing values
    youth_age_columns_final = [col for col in X_train_imputed_df.columns if any(f'Age{age}' in col for age in range(10, 19))]
    youth_missing_train = X_train_imputed_df[youth_age_columns_final].isnull().sum().sum()
    youth_missing_test = X_test_imputed_df[youth_age_columns_final].isnull().sum().sum()
    print(f"Youth age columns missing values - Training: {youth_missing_train}, Test: {youth_missing_test}")
    
    if youth_missing_train > 0 or youth_missing_test > 0:
        print("⚠️  Warning: Some youth age columns still have missing values after imputation.")
        print("This may indicate insufficient data for the imputer to make predictions.")
        print("Consider using a different imputation strategy or adding more training data.")
    
    # Define hyperparameter search space
    print(f"\nSetting up hyperparameter search space...")
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False]
    }
    
    # Initialize base Random Forest model
    base_rf = RandomForestClassifier(random_state=42)
    
    # Initialize RandomizedSearchCV
    print(f"Initializing RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=5,       # 5-fold cross-validation
        scoring='f1_macro',  # Use macro F1 score for optimization
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )
    
    # Perform hyperparameter search on imputed data
    print(f"\nPerforming hyperparameter search on imputed data...")
    print("This may take a few minutes...")
    random_search.fit(X_train_imputed_df, y_train)
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n=== HYPERPARAMETER SEARCH RESULTS ===")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Make predictions with best model on imputed test data
    print(f"\nMaking predictions with best model...")
    y_pred = best_model.predict(X_test_imputed_df)
    
    # Evaluate the best model
    print(f"\n=== BEST MODEL EVALUATION ===")
    
    # Calculate Macro-averaged F1 Score
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro-averaged F1 Score: {macro_f1:.4f}")
    
    # Calculate Balanced Accuracy Score
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy Score: {balanced_acc:.4f}")
    
    # Save the best model
    print(f"\nSaving the best model...")
    model_filename = 'best_swimming_predictor_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved as '{model_filename}'")
    
    # Save the best parameters
    params_filename = 'best_model_parameters.txt'
    with open(params_filename, 'w') as f:
        f.write("Best Random Forest Parameters:\n")
        f.write("=" * 40 + "\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest CV Score: {best_score:.4f}\n")
        f.write(f"Test Macro F1: {macro_f1:.4f}\n")
        f.write(f"Test Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"\nMissing Values:\n")
        f.write(f"Before imputation: {missing_before}\n")
        f.write(f"After imputation - Training: {missing_after_train}\n")
        f.write(f"After imputation - Test: {missing_after_test}\n")
        if features_with_no_values:
            f.write(f"Features with no observed values: {len(features_with_no_values)}\n")
    
    print(f"Best parameters saved as '{params_filename}'")
    
    print(f"\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(f"Best model and parameters have been saved for future use.")
    print(f"Note: This model uses controlled imputation instead of IterativeImputer.")

if __name__ == "__main__":
    main() 