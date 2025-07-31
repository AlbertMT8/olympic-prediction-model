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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
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
    """
    print("Creating Target_Olympic_Event column from Olympic event indicators...")
    
    # Find all Olympic event columns
    olympic_cols = [col for col in df.columns if col.startswith('Olympic_')]
    print(f"Found {len(olympic_cols)} Olympic event columns")
    
    # Create target column
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
    
    X_clean = X.fillna(0)
    missing_after = X_clean.isnull().sum().sum()
    print(f"Missing values after imputation: {missing_after}")
    
    # Split the data
    print(f"\nSplitting data (80% train, 20% test)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, 
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
    
    # Perform hyperparameter search
    print(f"\nPerforming hyperparameter search...")
    print("This may take a few minutes...")
    random_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n=== HYPERPARAMETER SEARCH RESULTS ===")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Make predictions with best model
    print(f"\nMaking predictions with best model...")
    y_pred = best_model.predict(X_test)
    
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
    
    print(f"Best parameters saved as '{params_filename}'")
    
    print(f"\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(f"Best model and parameters have been saved for future use.")

if __name__ == "__main__":
    main() 