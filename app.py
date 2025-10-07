import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import os

# Create Flask app
app = Flask(__name__)

# =============================================================================
# === Final Forecasting Showdown: Legacy RMLA Only ===
# Web Implementation
# =============================================================================

# --- Re-usable functions ---

def load_and_prepare_data(file_path):
    """Loads and cleans the base data."""
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        return None, None
    df['collection'] = pd.to_numeric(df['collection'], errors='coerce').fillna(0)
    df = df.dropna(subset=['date']).sort_values(by='date').reset_index(drop=True)
    return df

def get_historical_data(df, test_month, n_months):
    """Gets historical data for the Legacy RMLA model."""
    days_in_month = test_month.days_in_month
    if days_in_month <= 29:
        return None
    potential_history = df[(df['month_days'] == days_in_month) & (df['year_month'] < test_month)]
    recent_n = potential_history['year_month'].unique()[-n_months:]
    history = potential_history[potential_history['year_month'].isin(recent_n)]
    return history if not history.empty else None

def prepare_data(df):
    """Prepare data for Legacy Model"""
    if df is None:
        return None, None
    
    legacy_df = df.copy()
    legacy_df['year_month'] = legacy_df['date'].dt.to_period('M')
    legacy_df['month_days'] = legacy_df['date'].dt.days_in_month
    legacy_df['day_of_month'] = legacy_df['date'].dt.day
    monthly_totals = legacy_df.groupby('year_month')['collection'].sum()
    legacy_df['mtd'] = legacy_df.groupby('year_month')['collection'].cumsum()
    legacy_df['percent_complete'] = legacy_df.apply(
        lambda row: row['mtd'] / monthly_totals[row['year_month']]
        if monthly_totals.get(row['year_month'], 0) != 0 else 0,
        axis=1
    )
    return legacy_df, monthly_totals

# Load data once when the app starts
file_path = os.path.join(os.path.dirname(__file__), "collection_data.csv")
base_df = load_and_prepare_data(file_path)
legacy_df, monthly_totals = prepare_data(base_df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_forecast')
def live_forecast():
    return render_template('live_forecast.html')

@app.route('/backtest')
def backtest():
    return render_template('backtest.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/api/live_forecast', methods=['POST'])
def api_live_forecast():
    global base_df, legacy_df, monthly_totals
    
    if base_df is None or legacy_df is None or monthly_totals is None:
        return jsonify({'error': 'Data not available'}), 500
    
    try:
        data = request.get_json()
        
        # Get input values
        forecast_month_str = data.get('forecast_month')
        day_to_forecast = data.get('day_to_forecast')
        current_mtd = float(data.get('current_mtd'))
        n_months_to_use = int(data.get('n_months_to_use', 6))
        
        # Process forecast month
        forecast_month = pd.Period(forecast_month_str)
        
        # Get historical data
        historical_data = get_historical_data(legacy_df, forecast_month, n_months_to_use)
        if historical_data is None:
            return jsonify({'error': 'Cannot forecast: Not enough historical data.'}), 400
        
        # Convert day_to_forecast to int if it's not None
        if day_to_forecast is not None:
            day_to_forecast = int(day_to_forecast)
        else:
            return jsonify({'error': 'Day to forecast is required'}), 400
        
        # Main Forecast
        legacy_curve = historical_data.groupby('day_of_month')['percent_complete'].mean().sort_index()
        legacy_attainment = legacy_curve.get(day_to_forecast, 0)
        forecast = current_mtd / legacy_attainment if legacy_attainment > 0 else 0
        
        # Margin of Error Calculation
        errors = []
        for hist_month in historical_data['year_month'].unique():
            temp_hist_data = get_historical_data(legacy_df, hist_month, n_months_to_use)
            if temp_hist_data is None:
                continue
            
            temp_curve = temp_hist_data.groupby('day_of_month')['percent_complete'].mean()
            hist_attainment = temp_curve.get(day_to_forecast, 0)
            
            hist_month_day_data = legacy_df[
                (legacy_df['year_month'] == hist_month) & (legacy_df['day_of_month'] == day_to_forecast)
            ]
            if hist_month_day_data.empty:
                continue

            hist_mtd = hist_month_day_data['mtd'].iloc[0]
            hist_actual_total = monthly_totals[hist_month]
            
            hist_forecast = hist_mtd / hist_attainment if hist_attainment > 0 else 0
            error = abs(hist_forecast - hist_actual_total) / hist_actual_total if hist_actual_total > 0 else 0
            errors.append(error)
        
        avg_margin_of_error = np.mean(errors) if errors else 0

        return jsonify({
            'forecast': round(forecast, 2),
            'avg_margin_of_error': avg_margin_of_error
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    global base_df, legacy_df, monthly_totals
    
    if base_df is None or legacy_df is None or monthly_totals is None:
        return jsonify({'error': 'Data not available'}), 500
    
    try:
        data = request.get_json()
        
        # Get input values
        test_period_months = int(data.get('test_period_months', 6))
        n_for_rmla_model = int(data.get('n_for_rmla_model', 6))
        
        all_months_sorted = sorted(legacy_df['year_month'].unique())
        months_to_test = all_months_sorted[-test_period_months:]

        records = []
        for m in months_to_test:
            historical_data = get_historical_data(legacy_df, m, n_for_rmla_model)
            if historical_data is None:
                continue
            legacy_curve = historical_data.groupby('day_of_month')['percent_complete'].mean().sort_index()
            
            month_data = legacy_df[legacy_df['year_month'] == m]
            actual_eom_total = monthly_totals[m]

            for _, day_row in month_data.iterrows():
                day, mtd_actual = day_row['day_of_month'], day_row['mtd']
                pct_legacy = legacy_curve.get(day, 0)
                fc_legacy = mtd_actual / pct_legacy if pct_legacy > 0 else 0

                records.append({
                    "Month": str(m),
                    "date": day_row['date'].strftime('%Y-%m-%d'),
                    "day_of_month": day,
                    "actual_eom_total": actual_eom_total,
                    "fc_legacy": fc_legacy,
                })

        if records:
            report = pd.DataFrame(records).replace([np.inf, -np.inf], 0)
            report['error_fc_legacy'] = abs(
                (report['fc_legacy'] - report['actual_eom_total']) / report['actual_eom_total']
            )
            
            # Monthly Performance
            monthly_summary = report.groupby('Month')[['error_fc_legacy']].mean()
            
            # Overall Performance
            accuracy = report['error_fc_legacy'].mean()
            
            # Format results
            monthly_results = []
            for month, errors in monthly_summary.iterrows():
                monthly_results.append({
                    'month': month,
                    'error': errors['error_fc_legacy']
                })
            
            return jsonify({
                'monthly_results': monthly_results,
                'overall_accuracy': accuracy
            })
        else:
            return jsonify({'error': 'No records generated'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload', methods=['POST'])
def api_upload():
    global base_df, legacy_df, monthly_totals
    
    try:
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file:
            # Save file to replace collection_data.csv
            file_path = os.path.join(os.path.dirname(__file__), "collection_data.csv")
            uploaded_file.save(file_path)
            
            # Reload data
            base_df = load_and_prepare_data(file_path)
            legacy_df, monthly_totals = prepare_data(base_df)
            
            return jsonify({'message': 'Data updated successfully'}), 200
        else:
            return jsonify({'error': 'No file provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)