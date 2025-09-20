import pandas as pd
from datetime import timedelta
from meteostat import Stations, Daily
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

# --- Configuration ---
EXCEL_FILE_PATH = 'milk_data.xlsx' # IMPORTANT: Change this to your Excel file's name

# --- Part 1: Load and Reshape Data from Final 'Wide' Format ---

def load_and_reshape_data(file_path):
    """
    Loads a 'wide' format Excel file where states are in Column A,
    Column B is ignored, and dates start in Column C.
    """
    print("Step 1: Loading and reshaping data from final Excel format...")
    try:
        df_wide = pd.read_excel(file_path, header=0)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the name and path.")
        return None

    state_column = df_wide.iloc[:, 0]
    data_columns = df_wide.iloc[:, 2:]

    date_headers = []
    current_date = pd.to_datetime(data_columns.columns[0])
    
    for i in range(len(data_columns.columns)):
        date_headers.append(current_date)
        current_date += timedelta(days=7)
    
    data_columns.columns = date_headers
    df_corrected_wide = pd.concat([state_column, data_columns], axis=1)
    df_corrected_wide.rename(columns={df_corrected_wide.columns[0]: 'State'}, inplace=True)

    df_long = df_corrected_wide.melt(
        id_vars=['State'],      
        var_name='Date',        
        value_name='Milk Volume'
    )
    
    state_abbreviations_map = {
        'CA': 'California', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 
        'IN': 'Indiana', 'KY': 'Kentucky', 'MD': 'Maryland', 'ME': 'Maine', 
        'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'NC': 'North Carolina', 
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NY': 'New York', 'OH': 'Ohio', 
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'SD': 'South Dakota', 'UT': 'Utah', 
        'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 
        'WV': 'West Virginia'
    }

    df_long.dropna(subset=['Milk Volume'], inplace=True) 
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long['State'] = df_long['State'].map(state_abbreviations_map)
    df_long.dropna(subset=['State'], inplace=True)

    print("Data successfully reshaped.")
    return df_long

# --- Part 2: Fetching Weather Data ---

def get_weekly_weather(df):
    """
    Fetches historical weather data including humidity and calculates THI.
    """
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
        # ... (rest of the function is the same, truncated for brevity)
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

# --- Part 3: Main Model Training Workflow ---

df = load_and_reshape_data(EXCEL_FILE_PATH)

if df is not None and not df.empty:
    weather_df = get_weekly_weather(df)
    df = pd.concat([df, weather_df], axis=1)
    df.dropna(subset=['Milk Volume', 'State', 'Date', 'avg_temp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if not df.empty:
        print("Step 3: Engineering features...")
        week_series = df['Date'].dt.isocalendar().week
        
        # --- NEW: Create cyclical time features ---
        # This helps the model understand that week 52 is close to week 1
        df['week_sin'] = np.sin(2 * np.pi * week_series / 52.0)
        df['week_cos'] = np.cos(2 * np.pi * week_series / 52.0)
        
        df['month'] = df['Date'].dt.month
        df = pd.get_dummies(df, columns=['State'], prefix='state')

        y = df['Milk Volume']
        X = df.drop(columns=['Date', 'Milk Volume'])

        print("Step 4: Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- NEW: Hyperparameter Tuning ---
        print("Step 5: Finding the best XGBoost settings (Hyperparameter Tuning)...")

        # Define the range of settings to test
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Create the XGBoost model
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=10)
        
        # Set up the randomized search. n_iter controls how many combinations to try.
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50, # Try 50 different combinations
            scoring='neg_mean_absolute_error',
            n_jobs=-1, # Use all available CPU cores
            cv=3,      # Use 3-fold cross-validation
            verbose=1,
            random_state=42
        )
        
        # Run the search
        random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Best Parameters Found: {random_search.best_params_}")
        
        # The best model is automatically saved
        best_model = random_search.best_estimator_

        # --- Step 6: Evaluate the BEST Model ---
        print("\nStep 6: Evaluating the best model found...")
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Model Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print("--------------------------------\n")
    else:
        print("Dataframe is empty after processing. Check your Excel file and state maps.")
else:
    print("Could not load or reshape data. Exiting.")