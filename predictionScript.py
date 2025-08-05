#!/usr/bin/env python3
"""
Swimming Olympic Event Prediction Script

This script loads the trained Random Forest model and predicts which Olympic swimming event
a user has the best chance of competing in based on their input data.
"""

import joblib
import pandas as pd
import numpy as np
import os

def load_model():
    """
    Load the trained model from the saved file.
    """
    try:
        model = joblib.load('best_swimming_predictor_model.pkl')
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("❌ Error: 'best_swimming_predictor_model.pkl' not found!")
        print("Please run train_swimming_predictor.py first to train and save the model.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def get_user_input():
    """
    Get basic user information.
    """
    print("\n=== SWIMMING OLYMPIC EVENT PREDICTOR ===")
    print("Let's find out which Olympic swimming event you have the best chance in!\n")
    
    # Get basic information
    try:
        height = float(input("Enter your height in cm: "))
        weight = float(input("Enter your weight in kg: "))
        race = input("Enter your race/ethnicity (White/Black/Asian/Hispanic/Pacific_Islander/Native_American): ").strip()
        gender = input("Enter your gender (M/F): ").strip().upper()
        age = int(input("Enter your age: "))
        
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        # Encode gender
        sex_encoded = 1 if gender == 'M' else 0
        
        # Encode race
        race_encoded = {
            'white': {'White': 1, 'Black': 0, 'Asian': 0, 'Hispanic': 0, 'Pacific_Islander': 0, 'Native_American': 0},
            'black': {'White': 0, 'Black': 1, 'Asian': 0, 'Hispanic': 0, 'Pacific_Islander': 0, 'Native_American': 0},
            'asian': {'White': 0, 'Black': 0, 'Asian': 1, 'Hispanic': 0, 'Pacific_Islander': 0, 'Native_American': 0},
            'hispanic': {'White': 0, 'Black': 0, 'Asian': 0, 'Hispanic': 1, 'Pacific_Islander': 0, 'Native_American': 0},
            'pacific_islander': {'White': 0, 'Black': 0, 'Asian': 0, 'Hispanic': 0, 'Pacific_Islander': 1, 'Native_American': 0},
            'native_american': {'White': 0, 'Black': 0, 'Asian': 0, 'Hispanic': 0, 'Pacific_Islander': 0, 'Native_American': 1}
        }
        
        race_lower = race.lower().replace(' ', '_')
        if race_lower in race_encoded:
            ethnicity_data = race_encoded[race_lower]
        else:
            print("⚠️  Race not recognized, defaulting to White")
            ethnicity_data = race_encoded['white']
        
        return {
            'height_cm': height,
            'weight_kg': weight,
            'bmi': bmi,
            'age': age,
            'sex_encoded': sex_encoded,
            'ethnicity_data': ethnicity_data
        }
        
    except ValueError:
        print("❌ Error: Please enter valid numbers for height, weight, and age.")
        return None

def get_course_type():
    """
    Get the course type for swim times.
    """
    print("\n=== COURSE TYPE ===")
    print("What course type are your swim times from?")
    print("1. SCY (Short Course Yards - 25 yards)")
    print("2. SCM (Short Course Meters - 25 meters)")
    print("3. LCM (Long Course Meters - 50 meters)")
    
    while True:
        try:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == '1':
                return 'SCY'
            elif choice == '2':
                return 'SCM'
            elif choice == '3':
                return 'LCM'
            else:
                print("❌ Please enter 1, 2, or 3.")
        except:
            print("❌ Invalid input. Please enter 1, 2, or 3.")

def get_available_events(course_type, age):
    """
    Return a list of available swimming events for time input based on course type and age.
    """
    training_features = load_training_features()
    if training_features is None:
        return []
    
    available_events = []
    
    # Define the events we want to check for (using exact naming patterns from training data)
    # Note: Using standardized names without 'm' after distances (as created by training script)
    event_patterns = [
        ('50m Freestyle', f'50Freestyle_{course_type}_Age{age}_FastestTime'),
        ('100m Freestyle', f'100Freestyle_{course_type}_Age{age}_FastestTime'),
        ('200m Freestyle', f'200Freestyle_{course_type}_Age{age}_FastestTime'),
        ('400m Freestyle', f'400Freestyle_{course_type}_Age{age}_FastestTime'),
        ('800m Freestyle', f'800Freestyle_{course_type}_Age{age}_FastestTime'),
        ('1500m Freestyle', f'1500Freestyle_{course_type}_Age{age}_FastestTime'),
        ('50m Backstroke', f'50Backstroke_{course_type}_Age{age}_FastestTime'),
        ('100m Backstroke', f'100Backstroke_{course_type}_Age{age}_FastestTime'),
        ('200m Backstroke', f'200Backstroke_{course_type}_Age{age}_FastestTime'),
        ('50m Breaststroke', f'50Breaststroke_{course_type}_Age{age}_FastestTime'),
        ('100m Breaststroke', f'100Breaststroke_{course_type}_Age{age}_FastestTime'),
        ('200m Breaststroke', f'200Breaststroke_{course_type}_Age{age}_FastestTime'),
        ('50m Butterfly', f'50Butterfly_{course_type}_Age{age}_FastestTime'),
        ('100m Butterfly', f'100Butterfly_{course_type}_Age{age}_FastestTime'),
        ('200m Butterfly', f'200Butterfly_{course_type}_Age{age}_FastestTime'),
        ('100m IM', f'100IM_{course_type}_Age{age}_FastestTime'),
        ('200m IM', f'200IM_{course_type}_Age{age}_FastestTime'),
        ('400m IM', f'400IM_{course_type}_Age{age}_FastestTime'),
    ]
    
    # Check which events actually exist in the training data
    for event_name, feature_name in event_patterns:
        if feature_name in training_features:
            available_events.append((event_name, feature_name))
    
    return available_events

def get_swim_times(events, course_type, age):
    """
    Get swim times for selected events.
    """
    print(f"\n=== SWIM TIMES ({course_type}) ===")
    print(f"Enter your best times from {course_type} competitions at age {age}:")
    
    if not events:
        print(f"⚠️  No events found for age {age} in {course_type} course type.")
        print("This might be because:")
        print(f"   - Age {age} data is not available in the training dataset")
        print(f"   - {course_type} course data is not available for age {age}")
        print("   - The feature naming pattern is different")
        print("\nAvailable age ranges in the dataset: 10-33")
        print("Available course types: SCY, SCM, LCM")
        print("\nTry using a different age or course type.")
        return {}
    
    print(f"✅ Found {len(events)} events available for age {age} in {course_type}:")
    print("Select which events you want to enter times for:")
    
    # Display available events
    for i, (event_name, _) in enumerate(events, 1):
        print(f"{i}. {event_name}")
    
    print(f"\nEnter the numbers of events you want to input times for (comma-separated, e.g., 1,3,5):")
    print("Or press Enter to skip all events.")
    
    try:
        selection = input("Your selection: ").strip()
        if not selection:
            return {}
        
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_events = [events[i] for i in selected_indices if 0 <= i < len(events)]
        
        times = {}
        print(f"\nEnter your best {course_type} times for the selected events:")
        print("Format: MM:SS.SS (e.g., 1:30.45 for 1 minute 30.45 seconds)")
        
        for event_name, feature_name in selected_events:
            while True:
                try:
                    time_input = input(f"{event_name} (MM:SS.SS): ").strip()
                    if not time_input:
                        print("⚠️  Skipping this event")
                        break
                    
                    # Parse time input
                    if ':' in time_input:
                        minutes, seconds = time_input.split(':')
                        total_seconds = float(minutes) * 60 + float(seconds)
                    else:
                        total_seconds = float(time_input)
                    
                    times[feature_name] = total_seconds
                    break
                    
                except ValueError:
                    print("❌ Invalid format. Please use MM:SS.SS or just seconds.")
        
        return times
        
    except (ValueError, IndexError):
        print("❌ Invalid selection. Please enter valid numbers.")
        return {}

def load_training_features():
    """
    Load the exact feature names from the imputed training data.
    """
    try:
        # Load the imputed training data that the model was actually trained on
        df = pd.read_csv('X_train_imputed.csv')
        return list(df.columns)
    except Exception as e:
        print(f"❌ Error loading training features: {e}")
        print("Make sure you have run train_swimming_predictor.py first to create X_train_imputed.csv")
        return None

def create_feature_vector(user_data, swim_times, course_type):
    """
    Create a feature vector for prediction using exact feature names from training data.
    """
    # Load exact feature names from training data
    training_features = load_training_features()
    if training_features is None:
        return None
    
    # Create a dictionary with all features initialized to defaults
    features = {}
    
    # Initialize all features with default values (excluding Name and Target_Olympic_Event)
    for feature in training_features:
        if feature not in ['Name', 'Target_Olympic_Event']:
            if 'Olympic_' in feature:
                features[feature] = 0  # Default for Olympic event indicators
            elif 'FastestTime' in feature:
                features[feature] = 999.0  # Default slow time for swim times
            else:
                features[feature] = 0  # Default for other features
    
    # Override with user's basic information
    features['Height_cm'] = user_data['height_cm']
    features['Weight_kg'] = user_data['weight_kg']
    features['BMI'] = user_data['bmi']
    features['Sex_encoded'] = user_data['sex_encoded']
    features['Ethnicity_White'] = user_data['ethnicity_data']['White']
    features['Ethnicity_Black'] = user_data['ethnicity_data']['Black']
    features['Ethnicity_Asian'] = user_data['ethnicity_data']['Asian']
    features['Ethnicity_Hispanic'] = user_data['ethnicity_data']['Hispanic']
    features['Ethnicity_Pacific_Islander'] = user_data['ethnicity_data']['Pacific_Islander']
    features['Ethnicity_Native_American'] = user_data['ethnicity_data']['Native_American']
    
    # Override with user's swim times
    events = get_available_events(course_type, user_data['age'])
    for _, feature_name in events:
        if feature_name in swim_times:
            features[feature_name] = swim_times[feature_name]
    
    return features

def predict_olympic_event(model, features):
    """
    Make prediction using the trained model.
    """
    # Load training features to get the correct order
    training_features = load_training_features()
    if training_features is None:
        return None, 0.0
    
    # Create DataFrame with features in the exact same order as training data
    # Exclude Name and Target_Olympic_Event columns
    prediction_features = [f for f in training_features if f not in ['Name', 'Target_Olympic_Event']]
    
    # Create ordered feature vector
    ordered_features = []
    for feature in prediction_features:
        if feature in features:
            ordered_features.append(features[feature])
        else:
            # Fallback defaults
            if 'Olympic_' in feature:
                ordered_features.append(0)
            elif 'FastestTime' in feature:
                ordered_features.append(999.0)
            else:
                ordered_features.append(0)
    
    # Convert to DataFrame with correct column order
    df = pd.DataFrame([ordered_features], columns=prediction_features)
    
    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    # Get confidence
    max_prob = max(probabilities)
    
    return prediction, max_prob

def display_results(prediction, confidence):
    """
    Display the prediction results.
    """
    print("\n" + "="*50)
    print("🏊‍♂️ OLYMPIC EVENT PREDICTION RESULTS 🏊‍♀️")
    print("="*50)
    
    if prediction == 'No_Olympic_Event':
        print(f"\n❌ Prediction: {prediction}")
        print("💡 This means the model predicts you may not be competitive")
        print("   for Olympic swimming events based on your current data.")
        print("   However, this is just a prediction - keep training!")
    else:
        print(f"\n🎯 Predicted Olympic Event: {prediction}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        # Provide event-specific advice
        if 'Freestyle' in prediction:
            print("💡 Focus on freestyle technique and endurance training")
        elif 'Backstroke' in prediction:
            print("💡 Work on backstroke technique and core strength")
        elif 'Breaststroke' in prediction:
            print("💡 Develop breaststroke timing and leg strength")
        elif 'Butterfly' in prediction:
            print("💡 Build butterfly power and shoulder strength")
        elif 'IM' in prediction:
            print("💡 Train all four strokes and build endurance")
    
    print(f"\n⚠️  Disclaimer: This is a prediction based on limited data.")
    print("   Many factors affect Olympic qualification beyond just times.")
    print("   Use this as motivation, not a guarantee!")

def main():
    """
    Main function to run the prediction script.
    """
    # Load the trained model
    model = load_model()
    if model is None:
        return
    
    # Get user input
    user_data = get_user_input()
    if user_data is None:
        return
    
    # Get course type
    course_type = get_course_type()
    
    # Get available events and user selections
    events = get_available_events(course_type, user_data['age'])
    swim_times = get_swim_times(events, course_type, user_data['age'])
    
    # Create feature vector
    features = create_feature_vector(user_data, swim_times, course_type)
    if features is None:
        print("❌ Error: Could not create feature vector. Please check your training data file.")
        return
    
    # Make prediction
    prediction_result = predict_olympic_event(model, features)
    if prediction_result[0] is None:
        print("❌ Error: Could not make prediction. Please check your model and data.")
        return
    
    prediction, confidence = prediction_result
    
    # Display results
    display_results(prediction, confidence)
    
    print(f"\n🎉 Thanks for using the Olympic Swimming Event Predictor!")

if __name__ == "__main__":
    main()
