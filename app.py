import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from meteostat import Stations, Daily
import os
import google.generativeai as genai

print("Loading trained models...")
volume_model = joblib.load('models/milk_production_model.joblib')
volume_model_columns = joblib.load('models/model_columns.joblib')
fat_model = joblib.load('models/fat_percentage_model.joblib')
fat_model_columns = joblib.load('models/fat_model_columns.joblib')
print("All models loaded successfully.")

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini API key configured successfully.")
except KeyError:
    print("\n\n--- WARNING: GEMINI_API_KEY environment variable not found ---")
    print("AI reasoning will use a fallback message.")
    print("Set your API key to enable generative AI reasoning.\n\n")


app = Flask(__name__)

state_coords = {
    'California': (36.77, -119.41), 'Colorado': (39.55, -105.78), 'Iowa': (41.87, -93.09), 'Idaho': (44.06, -114.74),
    'Illinois': (40.63, -89.39), 'Indiana': (40.26, -86.13), 'Kentucky': (37.83, -84.27),
    'Maryland': (39.04, -76.64), 'Maine': (45.25, -69.44), 'Michigan': (44.31, -85.60),
    'Minnesota': (46.72, -94.68), 'Missouri': (38.57, -92.60), 'North Carolina': (35.75, -79.01),
    'New Hampshire': (43.19, -71.57), 'New Jersey': (40.05, -74.40), 'New York': (42.71, -74.91),
    'Ohio': (40.41, -82.90), 'Oregon': (43.80, -120.55), 'Pennsylvania': (40.96, -77.72),
    'South Dakota': (43.96, -99.90), 'Utah': (39.32, -111.09), 'Virginia': (37.43, -78.65),
    'Vermont': (44.55, -72.57), 'Washington': (47.75, -120.74), 'Wisconsin': (44.78, -89.82),
    'West Virginia': (38.59, -80.45)
}


def get_weather_forecast(state, target_date):
    if state not in state_coords: return {'avg_temp': 20, 'total_precip': 5, 'avg_rhum': 70, 'avg_thi': 68}
    start_date = target_date - timedelta(days=3)
    end_date = target_date + timedelta(days=3)
    lat, lon = state_coords[state]
    stations = Stations()
    nearby_stations = stations.nearby(lat, lon)
    station = nearby_stations.fetch(1)
    if station.empty: return {'avg_temp': 20, 'total_precip': 5, 'avg_rhum': 70, 'avg_thi': 68}
    station_id = station.index[0]
    data = Daily(station_id, start_date, end_date).fetch()
    if data.empty: return {'avg_temp': 20, 'total_precip': 5, 'avg_rhum': 70, 'avg_thi': 68}
    weather = {'avg_temp': np.nan, 'total_precip': np.nan, 'avg_rhum': np.nan, 'avg_thi': np.nan}
    if 'tavg' in data.columns: weather['avg_temp'] = data['tavg'].mean()
    if 'prcp' in data.columns: weather['total_precip'] = data['prcp'].sum()
    if 'rhum' in data.columns: weather['avg_rhum'] = data['rhum'].mean()
    if 'tavg' in data and 'rhum' in data and not data['tavg'].isnull().all() and not data['rhum'].isnull().all():
        temp_c, humidity_pct = data['tavg'], data['rhum']
        thi = temp_c - (0.55 - (0.55 * humidity_pct / 100)) * (temp_c - 14.5)
        weather['avg_thi'] = thi.mean()
    return weather

def run_fat_prediction(state, target_date, weather_data, milk_type):
    input_data = {col: [0.0] for col in fat_model_columns}
    input_df = pd.DataFrame.from_dict(input_data)
    for key, value in weather_data.items():
        if key in input_df.columns: input_df.loc[0, key] = value
    week_of_year = target_date.isocalendar().week
    input_df.loc[0, 'month'] = target_date.month
    input_df.loc[0, 'week_sin'] = np.sin(2 * np.pi * week_of_year / 52.0)
    input_df.loc[0, 'week_cos'] = np.cos(2 * np.pi * week_of_year / 52.0)
    state_col = f'state_{state}'
    if state_col in input_df.columns: input_df.loc[0, state_col] = 1
    milk_type_col = f"type_{milk_type.replace(' ', '_')}"
    if milk_type_col in input_df.columns: input_df.loc[0, milk_type_col] = 1
    prediction = fat_model.predict(input_df)[0]
    return float(prediction)

def run_volume_prediction(state, target_date, weather_data, predicted_fat_decimal, milk_type):
    input_data = {col: [0.0] for col in volume_model_columns}
    input_df = pd.DataFrame.from_dict(input_data)
    input_df.loc[0, 'Milk Fat Percentage'] = predicted_fat_decimal
    for key, value in weather_data.items():
        if key in input_df.columns: input_df.loc[0, key] = value
    week_of_year = target_date.isocalendar().week
    input_df.loc[0, 'month'] = target_date.month
    input_df.loc[0, 'week_sin'] = np.sin(2 * np.pi * week_of_year / 52.0)
    input_df.loc[0, 'week_cos'] = np.cos(2 * np.pi * week_of_year / 52.0)
    state_col = f'state_{state}'
    if state_col in input_df.columns: input_df.loc[0, state_col] = 1
    milk_type_col = f"type_{milk_type.replace(' ', '_')}"
    if milk_type_col in input_df.columns: input_df.loc[0, milk_type_col] = 1
    prediction = volume_model.predict(input_df)[0]
    return round(float(prediction))

def get_generative_reasoning(state, milk_type, volume, fat_percent, weather, date):
    """Builds a prompt and calls the Gemini API with a retry mechanism."""
    if "GEMINI_API_KEY" not in os.environ:
        return "Generative AI reasoning is unavailable. Please configure the GEMINI_API_KEY environment variable."

    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"As a dairy industry analyst, provide a brief, 6-7 sentence professional explanation(but also comperehendible to the normal man) for a milk forecast. Do not use markdown. Data: State={state}, Milk Type={milk_type}, Week of={date.strftime('%B %d, %Y')}, Predicted Volume={volume:,} lbs, Predicted Fat={fat_percent:.2f}%, Avg Temp={weather.get('avg_temp', 'N/A'):.1f}°C, Avg THI={weather.get('avg_thi', 'N/A'):.1f}. If THI > 72, mention heat stress. If THI < 72, mention favorable conditions."
    
    for i in range(3): # Try up to 3 times
        try:
            response = model.generate_content(prompt)
            if response.parts:
                return response.parts[0].text.strip()
            else:
                print(f"Warning: Gemini API returned an empty response (Attempt {i+1}/3).")
        except Exception as e:
            print(f"Error calling Gemini API (Attempt {i+1}/3): {e}")
            time.sleep(1) # Wait 1 second before retrying
            
    # If all retries fail, return a final error message
    return "The AI analyst is currently unavailable due to high traffic. Please try again in a moment."

# --- API Endpoint ---
@app.route('/predict')
def predict():
    state = request.args.get('state', 'Unknown')
    days = int(request.args.get('days', 7))
    milk_type = request.args.get('milk_type', 'Organic')
    target_date = datetime.now() + timedelta(days=days)
    weather_data = get_weather_forecast(state, target_date)
    predicted_fat_decimal = run_fat_prediction(state, target_date, weather_data, milk_type)
    predicted_volume = run_volume_prediction(state, target_date, weather_data, predicted_fat_decimal, milk_type)
    predicted_fat_display = predicted_fat_decimal * 100
    
    reasoning = get_generative_reasoning(
        state, milk_type, predicted_volume, predicted_fat_display, weather_data, target_date
    )

    return jsonify({
        "predicted_date": target_date.strftime('%B %d, %Y'),
        "estimate": f"{predicted_volume:,} lbs",
        "fat_percentage": f"{predicted_fat_display:.2f}%",
        "reasoning": reasoning
    })

@app.route('/')
def index():
    """Serves the main index.html file."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)