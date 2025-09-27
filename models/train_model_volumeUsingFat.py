import pandas as pd
from datetime import timedelta
from meteostat import Stations, Daily
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib


MILK_VOLUME_FILE = 'models/milk_data.xlsx'
FAT_PERCENTAGE_FILE = 'models/Fat_Percentage.xlsx'


def load_and_merge_data(volume_path, fat_path):
    """
    Loads both milk volume (with milk type) and fat percentage data,
    reshapes them, and merges them into a single DataFrame.
    """
    print("Step 1: Loading and reshaping data from both Excel files...")
    state_abbreviations_map = {
        'CA': 'California', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 
        'IN': 'Indiana', 'KY': 'Kentucky', 'MD': 'Maryland', 'ME': 'Maine', 
        'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'NC': 'North Carolina', 
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NY': 'New York', 'OH': 'Ohio', 
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'SD': 'South Dakota', 'UT': 'Utah', 
        'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 
        'WV': 'West Virginia'
    }

    
    def reshape_file(file_path, value_name, include_milk_type):
        try:
            df_wide = pd.read_excel(file_path, header=0)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None

        
        if include_milk_type:
            id_vars = ['State', 'Milk Type']
            id_columns = df_wide.iloc[:, [0, 1]]
            id_columns.columns = id_vars
            data_columns = df_wide.iloc[:, 2:]
        else:
            id_vars = ['State']
            id_columns = df_wide.iloc[:, [0]]
            id_columns.columns = id_vars
            
            data_columns = df_wide.iloc[:, 2:]

        date_headers = []
        current_date = pd.to_datetime(data_columns.columns[0])
        for i in range(len(data_columns.columns)):
            date_headers.append(current_date)
            current_date += timedelta(days=7)
        
        data_columns.columns = date_headers
        df_corrected_wide = pd.concat([id_columns, data_columns], axis=1)
        
        df_long = df_corrected_wide.melt(
            id_vars=id_vars,      
            var_name='Date',        
            value_name=value_name
        )
        
        df_long['State'] = df_long['State'].map(state_abbreviations_map)
        df_long.dropna(subset=[value_name, 'State'], inplace=True)
        return df_long

    df_volume = reshape_file(volume_path, 'Milk Volume', include_milk_type=True)

    df_fat = reshape_file(fat_path, 'Milk Fat Percentage', include_milk_type=False)

    if df_volume is None or df_fat is None:
        return None
        
    print("Merging the two datasets...")
    df_merged = pd.merge(df_volume, df_fat, on=['State', 'Date'], how='inner')

    df_merged['Date'] = pd.to_datetime(df_merged['Date'])
    df_merged.dropna(subset=['Milk Type'], inplace=True)

    print("Data successfully reshaped and merged.")
    return df_merged


def get_weekly_weather(df):
    print("Step 2: Starting to fetch weather data... this may take a few minutes.")
    weather_cache = {}
    weather_data_rows = []
    state_coords = {
        'California': (36.77, -119.41), 'Iowa': (41.87, -93.09), 'Idaho': (44.06, -114.74),
        'Illinois': (40.63, -89.39), 'Indiana': (40.26, -86.13), 'Kentucky': (37.83, -84.27),
        'Maryland': (39.04, -76.64), 'Maine': (45.25, -69.44), 'Michigan': (44.31, -85.60),
        'Minnesota': (46.72, -94.68), 'Missouri': (38.57, -92.60), 'North Carolina': (35.75, -79.01),
        'New Hampshire': (43.19, -71.57), 'New Jersey': (40.05, -74.40), 'New York': (42.71, -74.91),
        'Ohio': (40.41, -82.90), 'Oregon': (43.80, -120.55), 'Pennsylvania': (40.96, -77.72),
        'South Dakota': (43.96, -99.90), 'Utah': (39.32, -111.09), 'Virginia': (37.43, -78.65),
        'Vermont': (44.55, -72.57), 'Washington': (47.75, -120.74), 'Wisconsin': (44.78, -89.82),
        'West Virginia': (38.59, -80.45)
    }
    stations = Stations()
    for index, row in df.iterrows():
        state = row['State']
        week_start_date = row['Date']
        week_end_date = week_start_date + timedelta(days=6)
        cache_key = (state, week_start_date.year, week_start_date.isocalendar()[1])
        if cache_key in weather_cache:
            weather_summary = weather_cache[cache_key]
        else:
            weather_summary = {'avg_temp': np.nan, 'total_precip': np.nan, 'avg_rhum': np.nan, 'avg_thi': np.nan}
            if state in state_coords:
                lat, lon = state_coords[state]
                nearby_stations = stations.nearby(lat, lon)
                station = nearby_stations.fetch(1)
                if not station.empty:
                    station_id = station.index[0]
                    data = Daily(station_id, week_start_date, week_end_date).fetch()
                    if not data.empty:
                        if 'tavg' in data.columns:
                            weather_summary['avg_temp'] = data['tavg'].mean()
                        if 'prcp' in data.columns:
                            weather_summary['total_precip'] = data['prcp'].sum()
                        if 'rhum' in data.columns:
                            weather_summary['avg_rhum'] = data['rhum'].mean()
                        if 'tavg' in data.columns and 'rhum' in data.columns and not data['tavg'].isnull().all() and not data['rhum'].isnull().all():
                            temp_c = data['tavg']
                            humidity_pct = data['rhum']
                            thi = temp_c - (0.55 - (0.55 * humidity_pct / 100)) * (temp_c - 14.5)
                            weather_summary['avg_thi'] = thi.mean()
            weather_cache[cache_key] = weather_summary
        weather_data_rows.append(weather_summary)
        if (index + 1) % 10 == 0:
            print(f"  Processed {index + 1}/{len(df)} rows...")
    print("Weather data fetching complete.")
    return pd.DataFrame(weather_data_rows, index=df.index)


df = load_and_merge_data(MILK_VOLUME_FILE, FAT_PERCENTAGE_FILE)
if df is not None and not df.empty:
    weather_df = get_weekly_weather(df)
    df = pd.concat([df, weather_df], axis=1)
    
    df.dropna(subset=['Milk Volume', 'Milk Fat Percentage', 'State', 'Date', 'avg_temp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if not df.empty:
        print("Step 3: Engineering more features...")
        week_series = df['Date'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * week_series / 52.0)
        df['week_cos'] = np.cos(2 * np.pi * week_series / 52.0)
        df['month'] = df['Date'].dt.month

        # --- UPDATED: Encode 'Milk Type' along with 'State' ---
        df = pd.get_dummies(df, columns=['State', 'Milk Type'], prefix=['state', 'type'])

        y = df['Milk Volume']
        # The new 'type' columns are now automatically included as features in X
        X = df.drop(columns=['Date', 'Milk Volume'])

        print("Step 4: Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Step 5: Finding the best XGBoost settings (Hyperparameter Tuning)...")
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=10)
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            cv=3,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Best Parameters Found: {random_search.best_params_}")
        
        best_model = random_search.best_estimator_

        print("\nStep 6: Evaluating the best model found...")
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Model Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print("--------------------------------\n")
        
        
        print("--- Saving Model ---")
        joblib.dump(best_model, 'models/milk_production_model.joblib')
        print("Model saved to milk_production_model.joblib")

        model_columns = X.columns.tolist()
        joblib.dump(model_columns, 'models/model_columns.joblib')
        print("Model columns saved to model_columns.joblib")
        
    else:
        print("Dataframe is empty after processing. Check your Excel file and state maps.")
else:
    print("Could not load or reshape data. Exiting.")