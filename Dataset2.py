import pandas as pd
import json
import glob
from datetime import datetime, timedelta

# Load holidays from JSON files
def load_holidays(holiday_dir):
    holiday_info_dict = {}
    holiday_files = glob.glob(f'{holiday_dir}/*.json')
    for holiday_file in holiday_files:
        with open(holiday_file, 'r', encoding='utf-8') as file:
            holidays = json.load(file)['days']
            for item in holidays:
                holiday_info_dict[item['date']] = {'name': item['name']}
    return holiday_info_dict

# Add lagged features to the DataFrame
def add_lagged_features(df, column_name, lag_days=10):
    for day in range(1, lag_days + 1):
        df[f'{column_name}_lag_{day}'] = df[column_name].shift(day)
    return df

# Merge clinic, weather, and holiday data
def merge_data(clinic_data_path, weather_data_dir, holiday_dir, output_path):
    # Read clinic data
    clinic_df = pd.read_csv(clinic_data_path)

    # Process weather data
    weather_files = glob.glob(f'{weather_data_dir}/*.json')
    weather_data = []
    for file in weather_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for day in data['data']['weather']:
                weather_data.append({
                    'date': day['date'],
                    'maxtempC': day['maxtempC'],
                    'mintempC': day['mintempC'],
                    'weatherCode': day['hourly'][0]['weatherCode']
                })

    weather_df = pd.DataFrame(weather_data)
    
    # Load holiday data
    holiday_info_dict = load_holidays(holiday_dir)
    
    # Ensure date format consistency
    clinic_df['date'] = pd.to_datetime(clinic_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Merge datasets
    merged_df = pd.merge(clinic_df, weather_df, on='date')

    # Add holiday name
    merged_df['holiday_name'] = merged_df['date'].dt.strftime('%Y-%m-%d').apply(lambda x: holiday_info_dict[x]['name'] if x in holiday_info_dict else '')

    # One-hot encode weekdays
    merged_df['weekday'] = merged_df['date'].dt.weekday
    weekday_dummies = pd.get_dummies(merged_df['weekday'], prefix='weekday').astype(int)
    merged_df = pd.concat([merged_df, weekday_dummies], axis=1)
    
    # Add lagged DAILY_COUNT features
    merged_df = add_lagged_features(merged_df, 'DAILY_COUNT')

    # Add month, day, and year columns
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day'] = merged_df['date'].dt.day
    merged_df['year'] = merged_df['date'].dt.year

    # One-hot encode holiday names
    holiday_dummies = pd.get_dummies(merged_df['holiday_name'], prefix='holiday').astype(int)
    merged_df = pd.concat([merged_df, holiday_dummies], axis=1)

    # Remove unwanted columns
    merged_df.drop(columns=['weekday', 'holiday_name'], inplace=True)

    # Convert all columns to numeric, if possible
    for col in merged_df.columns.difference(['date']):
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Save the merged dataset
    merged_df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    clinic_data_path = 'data/DailyCount/DailyCount.csv'
    weather_data_dir = 'data/Weather'
    holiday_dir = 'data/Holiday'
    output_path = 'dataset.csv'
    merge_data(clinic_data_path, weather_data_dir, holiday_dir, output_path)
