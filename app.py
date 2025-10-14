import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

if getattr(sys, 'frozen', False):
    RESOURCE_ROOT = Path(sys._MEIPASS)
    WORKSPACE_ROOT = Path(sys.executable).resolve().parent
else:
    RESOURCE_ROOT = Path(__file__).resolve().parent
    WORKSPACE_ROOT = RESOURCE_ROOT

TEMPLATE_ROOT = RESOURCE_ROOT / 'templates'
STATIC_ROOT = RESOURCE_ROOT / 'static'
DATA_ROOT = WORKSPACE_ROOT / 'data'
DEFAULT_DATA_ROOT = RESOURCE_ROOT / 'data'
DATA_ROOT.mkdir(parents=True, exist_ok=True)

flask_kwargs = {'template_folder': str(TEMPLATE_ROOT)}
if STATIC_ROOT.exists():
    flask_kwargs['static_folder'] = str(STATIC_ROOT)

# Create Flask app
app = Flask(__name__, **flask_kwargs)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-me')

TREND_CONFIG = {
    'collection': {
        'label': 'Collection Trend',
        'metric_label': 'Collection',
        'value_column': 'collection',
        'file_name': 'collection_data.csv',
    },
    'releases': {
        'label': 'Releases Trend',
        'metric_label': 'Releases',
        'value_column': 'releases',
        'file_name': 'releases_data.csv',
        'segment_column': 'loan_type',
        'all_segment_label': 'All Loan Types',
    },
}

trend_store = {}

for trend_key, config in TREND_CONFIG.items():
    data_dir = DATA_ROOT / trend_key
    data_dir.mkdir(parents=True, exist_ok=True)
    config['data_dir'] = str(data_dir)
    file_path = data_dir / config['file_name']
    if DEFAULT_DATA_ROOT.exists():
        default_file = DEFAULT_DATA_ROOT / trend_key / config['file_name']
        if default_file.exists() and not file_path.exists():
            shutil.copy2(default_file, file_path)
    config['file_path'] = str(file_path)
    trend_store[trend_key] = {
        'segments': {},
        'segment_labels': {},
        'error': None,
    }

# =============================================================================
# === Final Forecasting Showdown: Legacy RMLA Only ===
# Web Implementation
# =============================================================================

# --- Re-usable functions ---

def load_and_prepare_data(file_path, value_column, segment_column=None):
    """Loads and cleans the base data for a given trend."""
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        return None, {}

    if value_column not in df.columns:
        raise ValueError(f"Expected column '{value_column}' in {file_path}")

    df = df.rename(columns={value_column: 'collection'})
    df['collection'] = pd.to_numeric(df['collection'], errors='coerce').fillna(0)
    df = df.dropna(subset=['date']).copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    segments = {}
    if segment_column and segment_column in df.columns:
        df[segment_column] = df[segment_column].fillna('Unspecified').astype(str)
        for segment_value, seg_df in df.groupby(segment_column):
            aggregated = seg_df.groupby('date', as_index=False)['collection'].sum()
            segments[segment_value] = aggregated

    overall = df.groupby('date', as_index=False)['collection'].sum()
    return overall, segments

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

def refresh_trend_data(trend_key):
    """Reloads source data for a trend into memory."""
    config = TREND_CONFIG[trend_key]
    context = trend_store.setdefault(trend_key, {})
    segment_column = config.get('segment_column')

    context['segments'] = {}
    context['segment_labels'] = {}
    context['error'] = None

    try:
        overall_df, segment_frames = load_and_prepare_data(
            config['file_path'],
            config['value_column'],
            segment_column,
        )
    except ValueError as exc:
        error_message = str(exc)
        context['segments']['__all__'] = {
            'base_df': None,
            'legacy_df': None,
            'monthly_totals': None,
            'latest_period': None,
            'error': error_message,
        }
        context['segment_labels']['__all__'] = config.get(
            'all_segment_label', config['metric_label']
        )
        context['error'] = error_message
        return

    if overall_df is None:
        error_message = 'Data file not found'
        context['segments']['__all__'] = {
            'base_df': None,
            'legacy_df': None,
            'monthly_totals': None,
            'latest_period': None,
            'error': error_message,
        }
        context['segment_labels']['__all__'] = config.get(
            'all_segment_label', config['metric_label']
        )
        context['error'] = error_message
        return

    def build_segment_entry(df):
        if df is None or df.empty:
            return {
                'base_df': df,
                'legacy_df': None,
                'monthly_totals': None,
                'latest_period': None,
                'error': 'Data not available',
            }

        legacy_df, monthly_totals = prepare_data(df)
        if legacy_df is None or legacy_df.empty or monthly_totals is None:
            return {
                'base_df': df,
                'legacy_df': None,
                'monthly_totals': None,
                'latest_period': None,
                'error': 'Data not available',
            }

        latest_period = str(legacy_df['year_month'].max()) if not legacy_df.empty else None
        return {
            'base_df': df,
            'legacy_df': legacy_df,
            'monthly_totals': monthly_totals,
            'latest_period': latest_period,
            'error': None,
        }

    overall_entry = build_segment_entry(overall_df)
    context['segments']['__all__'] = overall_entry
    context['segment_labels']['__all__'] = config.get(
        'all_segment_label', config['metric_label']
    )

    if overall_entry['error']:
        context['error'] = overall_entry['error']

    for segment_value, segment_df in sorted(segment_frames.items()):
        segment_key = str(segment_value)
        segment_entry = build_segment_entry(segment_df)
        context['segments'][segment_key] = segment_entry
        context['segment_labels'][segment_key] = segment_key



def get_trend_context(trend_key):
    return trend_store.get(trend_key, {})


def get_selected_trend_key():
    trend_key = session.get('selected_trend')
    if trend_key not in TREND_CONFIG:
        return None
    return trend_key


def get_available_segments(trend_key):
    context = get_trend_context(trend_key)
    return context.get('segments', {})


def get_selected_segment_key(trend_key):
    available_segments = get_available_segments(trend_key)
    if not available_segments:
        return '__all__'

    selected = session.get('selected_segment', '__all__')
    if selected not in available_segments:
        selected = '__all__'
        session['selected_segment'] = selected
    return selected


def get_segment_context(trend_key, segment_key=None):
    available_segments = get_available_segments(trend_key)
    if not available_segments:
        return None

    if segment_key is None:
        segment_key = get_selected_segment_key(trend_key)

    segment_context = available_segments.get(segment_key)
    if segment_context is None:
        segment_context = available_segments.get('__all__')
    return segment_context


for trend in TREND_CONFIG:
    refresh_trend_data(trend)


@app.context_processor
def inject_trend_context():
    trend_key = get_selected_trend_key()
    config = TREND_CONFIG.get(trend_key)
    segment_options = {}
    selected_segment_key = None
    selected_segment_label = None
    latest_period = None

    if config and trend_key is not None:
        context = get_trend_context(trend_key)
        segment_options = context.get('segment_labels', {}) or {
            '__all__': config['metric_label']
        }
        selected_segment_key = get_selected_segment_key(trend_key)
        selected_segment_label = segment_options.get(selected_segment_key)
        segment_context = get_segment_context(trend_key, selected_segment_key)
        if segment_context:
            latest_period = segment_context.get('latest_period')

    return {
        'selected_trend_key': trend_key,
        'selected_trend_label': config['label'] if config else None,
        'selected_trend_metric_label': config['metric_label'] if config else None,
        'selected_trend_latest_period': latest_period,
        'selected_segment_key': selected_segment_key,
        'selected_segment_label': selected_segment_label,
        'segment_options': segment_options,
        'trend_options': TREND_CONFIG,
    }

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/select_trend', methods=['POST'])
def select_trend():
    trend_key = request.form.get('trend')
    if trend_key not in TREND_CONFIG:
        return redirect(url_for('index'))

    session['selected_trend'] = trend_key
    session['selected_segment'] = '__all__'
    next_page = request.form.get('next')
    return redirect(next_page or url_for('live_forecast'))


@app.route('/clear_trend')
def clear_trend():
    session.pop('selected_trend', None)
    session.pop('selected_segment', None)
    return redirect(url_for('index'))

@app.route('/live_forecast')
def live_forecast():
    if get_selected_trend_key() is None:
        return redirect(url_for('index'))
    return render_template('live_forecast.html')

@app.route('/backtest')
def backtest():
    if get_selected_trend_key() is None:
        return redirect(url_for('index'))
    return render_template('backtest.html')

@app.route('/upload')
def upload():
    if get_selected_trend_key() is None:
        return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/select_segment', methods=['POST'])
def select_segment():
    trend_key = get_selected_trend_key()
    if trend_key is None:
        return redirect(url_for('index'))

    segment = request.form.get('segment', '__all__')
    available_segments = get_available_segments(trend_key)
    if segment not in available_segments:
        segment = '__all__'

    session['selected_segment'] = segment

    next_page = request.form.get('next') or request.headers.get('Referer') or url_for('live_forecast')
    if not next_page or not str(next_page).startswith('/'):
        next_page = url_for('live_forecast')
    return redirect(next_page)

@app.route('/api/live_forecast', methods=['POST'])
def api_live_forecast():
    trend_key = get_selected_trend_key()
    if trend_key is None:
        return jsonify({'error': 'Select a trend before forecasting.'}), 400

    context = get_trend_context(trend_key)
    if context.get('error'):
        return jsonify({'error': context['error']}), 400

    segment_context = get_segment_context(trend_key)
    if segment_context is None or segment_context.get('error'):
        error_msg = segment_context.get('error') if segment_context else 'Data not available for the selected segment.'
        return jsonify({'error': error_msg}), 400

    legacy_df = segment_context.get('legacy_df')
    monthly_totals = segment_context.get('monthly_totals')
    
    if legacy_df is None or monthly_totals is None:
        return jsonify({'error': 'Data not available for the selected segment.'}), 500
    
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

        history_rows = historical_data[historical_data['day_of_month'] == day_to_forecast]
        history_details = []
        for _, hist_row in history_rows.iterrows():
            hist_month = hist_row['year_month']
            implied_forecast = (
                hist_row['mtd'] / hist_row['percent_complete']
                if hist_row['percent_complete'] > 0 else 0
            )
            actual_total = float(monthly_totals.get(hist_month, 0))
            margin_of_error = (
                abs(implied_forecast - actual_total) / actual_total
                if actual_total > 0 else 0
            )
            history_details.append({
                'source_month': str(hist_month),
                'percent_complete': float(hist_row['percent_complete']),
                'mtd': float(hist_row['mtd']),
                'actual_total': actual_total,
                'implied_forecast': float(implied_forecast),
                'margin_of_error': float(margin_of_error),
            })

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
            'avg_margin_of_error': avg_margin_of_error,
            'avg_percent_complete': legacy_attainment,
            'history': history_details,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    trend_key = get_selected_trend_key()
    if trend_key is None:
        return jsonify({'error': 'Select a trend before running the backtest.'}), 400

    context = get_trend_context(trend_key)
    if context.get('error'):
        return jsonify({'error': context['error']}), 400

    segment_context = get_segment_context(trend_key)
    if segment_context is None or segment_context.get('error'):
        error_msg = segment_context.get('error') if segment_context else 'Data not available for the selected segment.'
        return jsonify({'error': error_msg}), 400

    legacy_df = segment_context.get('legacy_df')
    monthly_totals = segment_context.get('monthly_totals')
    
    if legacy_df is None or monthly_totals is None:
        return jsonify({'error': 'Data not available for the selected segment.'}), 500
    
    try:
        data = request.get_json()
        
        # Get input values
        test_period_months = int(data.get('test_period_months', 6))
        n_for_rmla_model = int(data.get('n_for_rmla_model', 6))
        
        all_months_sorted = sorted(legacy_df['year_month'].unique())
        months_to_test = all_months_sorted[-test_period_months:]

        records = []
        history_payloads = []
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

                history_rows = historical_data[historical_data['day_of_month'] == day]
                history_details = []
                for _, hist_row in history_rows.iterrows():
                    hist_month = hist_row['year_month']
                    implied_forecast = (
                        hist_row['mtd'] / hist_row['percent_complete']
                        if hist_row['percent_complete'] > 0 else 0
                    )
                    actual_total = float(monthly_totals.get(hist_month, 0))
                    margin_of_error = (
                        abs(implied_forecast - actual_total) / actual_total
                        if actual_total > 0 else 0
                    )
                    history_details.append({
                        'source_month': str(hist_month),
                        'percent_complete': float(hist_row['percent_complete']),
                        'mtd': float(hist_row['mtd']),
                        'actual_total': actual_total,
                        'implied_forecast': float(implied_forecast),
                        'margin_of_error': float(margin_of_error),
                    })

                records.append({
                    "Month": str(m),
                    "date": day_row['date'].strftime('%Y-%m-%d'),
                    "day_of_month": day,
                    "actual_eom_total": actual_eom_total,
                    "fc_legacy": fc_legacy,
                    "mtd_actual": mtd_actual,
                    "avg_percent_complete": pct_legacy,
                })
                history_payloads.append(history_details)

        if records:
            report = pd.DataFrame(records).replace([np.inf, -np.inf], 0)
            report['history_details'] = history_payloads
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

            # Prepare day-level detail so the UI can inspect the raw backtest points
            daily_report = report.sort_values(['Month', 'day_of_month'])
            daily_results = []
            for _, row in daily_report.iterrows():
                avg_pct = float(row.get('avg_percent_complete', 0) or 0)
                daily_results.append({
                    'month': row['Month'],
                    'date': row['date'],
                    'day_of_month': int(row['day_of_month']),
                    'forecast': float(row['fc_legacy']),
                    'actual_total': float(row['actual_eom_total']),
                    'mtd': float(row['mtd_actual']),
                    'avg_percent_complete': avg_pct,
                    'error': float(row['error_fc_legacy']),
                    'history': row.get('history_details', []),
                })

            return jsonify({
                'monthly_results': monthly_results,
                'daily_results': daily_results,
                'overall_accuracy': accuracy
            })
        else:
            return jsonify({'error': 'No records generated'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload', methods=['POST'])
def api_upload():
    trend_key = get_selected_trend_key()
    if trend_key is None:
        return jsonify({'error': 'Select a trend before uploading data.'}), 400

    config = TREND_CONFIG[trend_key]

    try:
        uploaded_file = request.files['file']

        if not uploaded_file:
            return jsonify({'error': 'No file provided'}), 400

        file_path = config['file_path']
        uploaded_file.save(file_path)

        refresh_trend_data(trend_key)
        context = get_trend_context(trend_key)
        if context.get('error'):
            return jsonify({'error': context['error']}), 400

        # Ensure the selected segment is still valid after refresh
        get_selected_segment_key(trend_key)

        return jsonify({'message': f"{config['label']} data updated successfully"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
