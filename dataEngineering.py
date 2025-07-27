import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict

def parse_race_date(date_str: str) -> str:
    """Parse race date from various formats to YYYY-MM-DD."""
    if not date_str or pd.isna(date_str):
        return None
    
    try:
        # Handle "Apr 21–24, 2025" format - take the first date
        if '–' in date_str or '-' in date_str:
            # Extract the first date from range
            first_date_match = re.search(r'([A-Za-z]+)\s+(\d{1,2})[–\-]', date_str)
            year_match = re.search(r'(\d{4})', date_str)
            
            if first_date_match and year_match:
                month_str = first_date_match.group(1)
                day = first_date_match.group(2)
                year = year_match.group(1)
                
                # Convert month name to number
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                
                month = month_map.get(month_str[:3], '01')
                return f"{year}-{month}-{day.zfill(2)}"
        
        # Handle other date formats
        return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d') if pd.notna(pd.to_datetime(date_str, errors='coerce')) else None
        
    except:
        return None

def normalize_event_name(event_name: str) -> str:
    """Normalize event names to consistent format for column naming."""
    if not event_name:
        return ""
    
    # Convert to lowercase and remove extra spaces
    event = event_name.lower().strip()
    
    # Handle "400 L Free" format (L = Long Course)
    event = event.replace(' l ', ' ').replace(' s ', ' ')
    
    # Extract distance and stroke
    distance_match = re.search(r'(\d+)\s*(?:meter|m)?', event)
    stroke_match = re.search(r'(free|back|breast|fly|im|medley)', event)
    
    if distance_match and stroke_match:
        distance = distance_match.group(1)
        stroke = stroke_match.group(1)
        
        # Map stroke abbreviations to column format
        stroke_map = {
            'free': 'Free',
            'back': 'Back', 
            'breast': 'Breast',
            'fly': 'Fly',
            'im': 'IM',
            'medley': 'IM'
        }
        
        return f"{distance}m{stroke_map.get(stroke, stroke.title())}"
    
    return event_name

def calculate_age_at_race(birth_date, race_date):
    """Calculate age at race date."""
    try:
        if pd.isna(birth_date) or pd.isna(race_date):
            return None
        
        birth = pd.to_datetime(birth_date)
        race = pd.to_datetime(race_date)
        
        age = race.year - birth.year
        if race.month < birth.month or (race.month == birth.month and race.day < birth.day):
            age -= 1
        return max(0, age)  # Ensure non-negative age
    except:
        return None

def is_individual_event(event_name: str) -> bool:
    """Check if event is individual (not relay or open water)."""
    if not event_name:
        return False
    
    event_lower = event_name.lower()
    
    # Exclude relays
    if 'relay' in event_lower or '4x' in event_lower:
        return False
    
    # Exclude open water/marathon
    if 'marathon' in event_lower or 'open water' in event_lower or '10k' in event_lower:
        return False
    
    return True

def get_olympic_events() -> List[str]:
    """Return list of standard Olympic swimming events (LCM only)."""
    return [
        # Freestyle
        "50m Freestyle", "100m Freestyle", "200m Freestyle", "400m Freestyle", 
        "800m Freestyle", "1500m Freestyle",
        # Backstroke
        "100m Backstroke", "200m Backstroke",
        # Breaststroke  
        "100m Breaststroke", "200m Breaststroke",
        # Butterfly
        "100m Butterfly", "200m Butterfly",
        # Individual Medley
        "200m IM", "400m IM",
        # Relays
        "4x100m Freestyle Relay", "4x200m Freestyle Relay", "4x100m Medley Relay",
        "4x100m Mixed Medley Relay"
    ]

def get_individual_event_combinations() -> List[tuple]:
    """Return list of all individual event-course combinations (61 total)."""
    events = [
        # Freestyle (9 events)
        ("50", "Freestyle"),
        ("100", "Freestyle"), 
        ("200", "Freestyle"),
        ("400", "Freestyle"),
        ("500", "Freestyle"),
        ("800", "Freestyle"),
        ("1000", "Freestyle"),
        ("1500", "Freestyle"),
        ("1650", "Freestyle"),
        # Backstroke (3 events)
        ("50", "Backstroke"),
        ("100", "Backstroke"),
        ("200", "Backstroke"),
        # Breaststroke (3 events)
        ("50", "Breaststroke"),
        ("100", "Breaststroke"),
        ("200", "Breaststroke"),
        # Butterfly (3 events)
        ("50", "Butterfly"),
        ("100", "Butterfly"),
        ("200", "Butterfly"),
        # Individual Medley (3 events)
        ("100", "IM"),
        ("200", "IM"),
        ("400", "IM")
    ]
    
    courses = ["SCY", "SCM", "LCM"]
    combinations = []
    
    for distance, stroke in events:
        for course in courses:
            combinations.append((distance, stroke, course))
    
    return combinations

def process_swimmer_data(results: List[Dict], all_scraped_race_records: List[Dict]) -> pd.DataFrame:
    """
    Process two lists of dictionaries into a comprehensive DataFrame.
    
    Args:
        results: List of swimmer static data dictionaries
        all_scraped_race_records: List of race record dictionaries
    
    Returns:
        DataFrame with one row per swimmer and all specified columns
    """
    
    # Step 1: Convert to DataFrames
    df_static = pd.DataFrame(results)
    df_races = pd.DataFrame(all_scraped_race_records)
    
    print(f"Loaded {len(df_static)} static records and {len(df_races)} race records")
    
    # Step 2: Data Cleaning on df_static
    if 'Swimmer_ID' in df_static.columns:
        df_static = df_static.rename(columns={'Swimmer_ID': 'Name'})
    elif 'name' in df_static.columns:
        df_static = df_static.rename(columns={'name': 'Name'})
    
    # Convert Date_of_Birth to datetime
    if 'Date_of_Birth' in df_static.columns:
        df_static['Date_of_Birth'] = pd.to_datetime(df_static['Date_of_Birth'], errors='coerce')
    elif 'dob' in df_static.columns:
        df_static['Date_of_Birth'] = pd.to_datetime(df_static['dob'], errors='coerce')
    
    # Handle Sex column
    if 'Sex' not in df_static.columns and 'gender' in df_static.columns:
        df_static['Sex'] = df_static['gender']
    
    # Handle Height/Weight columns
    if 'Height_cm' not in df_static.columns and 'height_cm' in df_static.columns:
        df_static['Height_cm'] = df_static['height_cm']
    if 'Weight_kg' not in df_static.columns and 'weight_kg' in df_static.columns:
        df_static['Weight_kg'] = df_static['weight_kg']
    
    # Handle Ethnicity column
    if 'Ethnicity' not in df_static.columns and 'race' in df_static.columns:
        df_static['Ethnicity'] = df_static['race']
    
    # Step 3: Data Cleaning on df_races
    if 'Swimmer_ID' in df_races.columns:
        df_races = df_races.rename(columns={'Swimmer_ID': 'Name'})
    
    # Convert Race_Date to datetime
    df_races['Race_Date'] = df_races['Race_Date'].apply(parse_race_date)
    
    # Step 4: Merge static and race data for age calculation
    df_merged = df_races.merge(df_static[['Name', 'Date_of_Birth']], on='Name', how='left')
    
    # Calculate Age_at_Race
    df_merged['Age_at_Race'] = df_merged.apply(
        lambda row: calculate_age_at_race(row['Date_of_Birth'], row['Race_Date']), axis=1
    )
    
    # Normalize Event_Name
    df_merged['Normalized_Event_Name'] = df_merged['Event_Name'].apply(normalize_event_name)
    
    # Filter to individual events only and ALL ages (not just youth)
    df_individual = df_merged[
        (df_merged['Event_Name'].apply(is_individual_event))
    ].copy()
    
    print(f"Filtered to {len(df_individual)} individual event records")
    
    # Step 5: Engineer Individual Event Fastest Times (ALL AGES, not just youth)
    # Group by swimmer, event, course, and age to find fastest times
    fastest_times = df_individual.groupby(['Name', 'Normalized_Event_Name', 'Course', 'Age_at_Race'])['Time_Seconds'].min().reset_index()
    
    # Create column names for fastest times
    fastest_times['Column_Name'] = fastest_times.apply(
        lambda row: f"{row['Normalized_Event_Name']}_{row['Course']}_Age{int(row['Age_at_Race'])}_FastestTime", 
        axis=1
    )
    
    # Pivot to wide format
    event_times_pivot = fastest_times.pivot(index='Name', columns='Column_Name', values='Time_Seconds')
    
    print(f"Created {len(event_times_pivot.columns)} event time columns from actual data")
    
    # Step 6: Create columns for the actual data we have (not all possible combinations)
    # The pivot table already contains the actual data columns
    print(f"Actual data columns created: {len(event_times_pivot.columns)}")
    print(f"Actual data points: {event_times_pivot.notna().sum().sum()}")
    
    # Get the actual column names from the data
    actual_columns = list(event_times_pivot.columns)
    
    # Create youth columns (10-18) with NaN values for the structure requested
    event_combinations = get_individual_event_combinations()
    youth_ages = list(range(10, 19))  # 10 through 18
    
    # Create all possible youth column names
    all_youth_columns = []
    for distance, stroke, course in event_combinations:
        for age in youth_ages:
            event_name = f"{distance}m{stroke}"
            col_name = f"{event_name}_{course}_Age{age}_FastestTime"
            all_youth_columns.append(col_name)
    
    print(f"Target youth columns: {len(all_youth_columns)} (21 events × 3 courses × 9 ages)")
    
    # Add missing youth columns with NaN values
    for col in all_youth_columns:
        if col not in event_times_pivot.columns:
            event_times_pivot[col] = np.nan
    
    # Ensure all columns are in the correct order (youth first, then actual data)
    all_columns = all_youth_columns + actual_columns
    event_times_pivot = event_times_pivot.reindex(columns=all_columns)
    
    print(f"Final youth event columns: {len(all_youth_columns)}")
    print(f"Final actual data columns: {len(actual_columns)}")
    print(f"Total event columns: {len(event_times_pivot.columns)}")
    print(f"Note: Youth columns (ages 10-18) will likely be empty since these are adult Olympic swimmers")
    print(f"Actual data columns contain the real race data for ages 18-33")
    print(f"Actual data contains ages: {sorted(df_individual['Age_at_Race'].dropna().unique())}")
    
    # Step 7: Merge with static data
    df_final = df_static.set_index('Name').join(event_times_pivot, how='left').reset_index()
    
    # Step 8: Encode Sex
    df_final['Sex_encoded'] = df_final['Sex'].map({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})
    
    # Step 9: Calculate BMI
    df_final['BMI'] = df_final.apply(
        lambda row: row['Weight_kg'] / ((row['Height_cm'] / 100) ** 2) 
        if pd.notna(row['Weight_kg']) and pd.notna(row['Height_cm']) and row['Height_cm'] > 0 
        else np.nan, 
        axis=1
    )
    
    # Step 10: One-Hot Encode Ethnicity
    ethnicity_columns = ['Ethnicity_White', 'Ethnicity_Black', 'Ethnicity_Asian', 
                        'Ethnicity_Hispanic', 'Ethnicity_Pacific_Islander', 'Ethnicity_Native_American']
    
    for col in ethnicity_columns:
        df_final[col] = 0
    
    # Map ethnicity values to binary columns
    ethnicity_mapping = {
        'white': 'Ethnicity_White',
        'black': 'Ethnicity_Black', 
        'asian': 'Ethnicity_Asian',
        'hispanic': 'Ethnicity_Hispanic',
        'pacific islander': 'Ethnicity_Pacific_Islander',
        'native american': 'Ethnicity_Native_American'
    }
    
    for idx, row in df_final.iterrows():
        ethnicity = str(row.get('Ethnicity', '')).lower() if pd.notna(row.get('Ethnicity')) else ''
        if ethnicity in ethnicity_mapping and ethnicity != 'unknown':
            df_final.at[idx, ethnicity_mapping[ethnicity]] = 1
    
    # Step 11: Engineer Olympic Event Columns
    olympic_events = get_olympic_events()
    
    # Create Olympic event columns
    for event in olympic_events:
        col_name = f"Olympic_{event.replace(' ', '_').replace('m', 'm_')}"
        df_final[col_name] = 0
    
    # Check participation in Olympic events
    olympic_participation = df_merged[
        (df_merged['Course'] == 'LCM') & 
        (df_merged['Event_Name'].str.contains('|'.join(olympic_events), case=False, na=False))
    ].groupby('Name')['Event_Name'].apply(list).reset_index()
    
    for idx, row in olympic_participation.iterrows():
        swimmer_name = row['Name']
        events = row['Event_Name']
        
        for event in events:
            for olympic_event in olympic_events:
                if olympic_event.lower() in event.lower():
                    col_name = f"Olympic_{olympic_event.replace(' ', '_').replace('m', 'm_')}"
                    if col_name in df_final.columns:
                        df_final.loc[df_final['Name'] == swimmer_name, col_name] = 1
    
    # Step 12: Final DataFrame Structure - REORDERED AS REQUESTED
    # 1. Name column first
    base_columns = ['Name']
    
    # 2. Sex_encoded
    # 3. 6 ethnicity binary columns
    demographic_columns = ['Sex_encoded']
    ethnicity_columns = ['Ethnicity_White', 'Ethnicity_Black', 'Ethnicity_Asian', 
                        'Ethnicity_Hispanic', 'Ethnicity_Pacific_Islander', 'Ethnicity_Native_American']
    
    # 4. Olympic event columns (sorted alphabetically)
    olympic_columns = sorted([col for col in df_final.columns if col.startswith('Olympic_')])
    
    # 5. Youth event time columns (567 columns) - sorted alphabetically - LAST
    youth_event_columns = sorted([col for col in df_final.columns if 'FastestTime' in col])
    
    # 6. Height_cm, Weight_kg, BMI (keeping these with other demographic data)
    additional_demographic_columns = ['Height_cm', 'Weight_kg', 'BMI']
    
    # Combine all columns in the new specified order
    final_columns = base_columns + demographic_columns + ethnicity_columns + olympic_columns + youth_event_columns + additional_demographic_columns
    
    # Select only the columns that exist in the DataFrame
    existing_columns = [col for col in final_columns if col in df_final.columns]
    
    df_final = df_final[existing_columns]
    
    print(f"Final DataFrame shape: {df_final.shape}")
    print(f"Youth event columns: {len(youth_event_columns)}")
    print(f"Olympic event columns: {len(olympic_columns)}")
    print(f"Total columns: {len(df_final.columns)}")
    
    return df_final

# Example usage and dummy data
if __name__ == "__main__":
    import json
    
    # Load actual data from JSON files
    print("Loading swimmer biometrics data...")
    with open('swimmer_biometrics.json', 'r') as f:
        results = json.load(f)
    
    print("Loading youth times data...")
    with open('olympic_swimmers_youth_times.json', 'r') as f:
        all_scraped_race_records = json.load(f)
    
    print(f"Loaded {len(results)} swimmers from biometrics data")
    print(f"Loaded {len(all_scraped_race_records)} race records from youth times data")
    
    # Process the data
    df_final = process_swimmer_data(results, all_scraped_race_records)
    
    # Save to CSV
    df_final.to_csv('final_swimmer_data.csv', index=False)
    
    print(f"\n✅ Data processing completed!")
    print(f"📊 Final DataFrame saved to: final_swimmer_data.csv")
    print(f"📈 Shape: {df_final.shape}")
    print(f"📋 Columns: {list(df_final.columns)}")
