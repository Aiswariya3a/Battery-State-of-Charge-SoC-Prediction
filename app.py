from flask import Flask, render_template
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random

app = Flask(__name__)

# Configuration
MODEL_FILE = 'model.json'
DATA_FILE = 'processed_dataset.csv'
SCALER_FILE = 'scaler.save'

required_files = {
    'Model': MODEL_FILE,
    'Data': DATA_FILE,
    'Scaler': SCALER_FILE
}

missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
if missing_files:
    raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

try:
    # Load model
    XGBoost_model_loaded = xgb.XGBRegressor()
    XGBoost_model_loaded.load_model(MODEL_FILE)
    
    # Load scaler
    scalerSummerTrips = joblib.load(SCALER_FILE)
except Exception as e:
    raise RuntimeError(f"Initialization failed: {str(e)}")

expected_features = [
    'avrg_Velocity [km/h]', 'avrg_Throttle [%]',
    'avrg_Motor Torque [Nm]', 'avrg_Longitudinal Acceleration [m/s^2]',
    'avrg_Regenerative Braking Signal ', 'avrg_Battery Voltage [V]',
    'avrg_Battery Current [A]', 'avrg_Battery Temperature [°C]',
    'avrg_Heating Power CAN [kW]', 'avrg_Requested Heating Power [W]',
    'avrg_AirCon Power [kW]', 'avrg_Heater Signal',
    'avrg_Ambient Temperature [°C]', 'avrg_Heat Exchanger Temperature [°C]',
    'avrg_Cabin Temperature Sensor [°C]', 'Elevation change', 'Previous SoC'
]

def load_dataset():
    try:
        df = pd.read_csv(DATA_FILE)
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = np.nan
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=expected_features)

@app.route('/')
def predict_random_row():
    df = load_dataset()
    
    if df.empty:
        return render_template('error.html', error="Dataset is empty")
    
    # Select a random row
    row_index = random.randint(0, len(df)-1)
    selected_row = df.iloc[row_index].to_dict()
    
    try:
        # Prepare input data
        input_data = {feature: selected_row[feature] for feature in expected_features}
        input_df = pd.DataFrame([input_data], columns=expected_features)
        
        # Make prediction
        input_scaled = scalerSummerTrips.transform(input_df)
        prediction = XGBoost_model_loaded.predict(input_scaled)[0]
        
        return render_template('dashboard.html', 
                            input_data=input_data,
                            prediction=prediction,
                            previous_soc=input_data['Previous SoC'],
                            row_index=row_index,
                            total_rows=len(df))
    except Exception as e:
        return render_template('error.html', 
                            error=f"Error processing row {row_index}: {str(e)}")

if __name__ == '__main__':
    print("Starting application...")
    
    app.run(debug=True)