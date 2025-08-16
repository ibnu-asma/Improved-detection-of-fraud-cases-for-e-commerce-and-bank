from flask import Flask, jsonify
import pandas as pd
from logger import SetupLogger

app = Flask(__name__)

# Initialize logger
logger = SetupLogger(log_file='logs/flask_app.log').get_logger()

# Load fraud data from CSV
def load_fraud_data():
    try:
        logger.info('Loading fraud data from CSV...')
        data = pd.read_csv('../data/merged_fraud_data.csv')
        logger.info('Fraud data loaded successfully.')
        return data
    except Exception as e:
        logger.error(f'Error loading fraud data: {e}')
        return None

# Endpoint to serve summary statistics
@app.route('/api/fraud_summary', methods=['GET'])
def fraud_summary():
    logger.info('Accessing /fraud_summary endpoint.')
    data = load_fraud_data()

    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    total_transactions = len(data)
    total_fraud_cases = len(data[data['class'] == 1])
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }
    logger.info(f'Successfully generated fraud summary: {summary}')
    return jsonify(summary)

# Endpoint to serve fraud trends
@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    logger.info('Accessing /fraud_trends endpoint.')
    data = load_fraud_data()

    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    data['purchase_date'] = pd.to_datetime(data['purchase_time']).dt.to_period('M').dt.strftime('%b %Y')
   

    trend_data = data.groupby('purchase_date').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()

    trend_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)

    logger.info('Successfully generated fraud trends.')
    return jsonify(trend_data.to_dict(orient='records'))

# Endpoint to serve fraud by location
@app.route('/api/fraud_by_location', methods=['GET'])
def fraud_by_location():
    logger.info('Accessing /fraud_by_location endpoint.')
    data = load_fraud_data()

    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    location_data = data.groupby('country').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()

    location_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)

    logger.info('Successfully generated fraud by location data.')
    return jsonify(location_data.to_dict(orient='records'))

# Endpoint to serve fraud cases by top device and browser combinations
@app.route('/api/fraud_by_device_browser', methods=['GET'])
def fraud_by_device_browser():
    logger.info('Accessing /fraud_by_device_browser endpoint.')
    data = load_fraud_data()

    if data is None:
        return jsonify({'error': 'Data could not be loaded.'}), 500

    # Filter for fraudulent cases only
    fraud_data = data[data['class'] == 1]

    # Group by device and browser, count fraud cases
    device_browser_data = (
        fraud_data.groupby(['device_id', 'browser'])
        .size()
        .reset_index(name='fraud_cases')
    )

    # Filter to include only the top 10 device-browser combinations with the most fraud cases
    top_device_browser_data = device_browser_data.nlargest(10, 'fraud_cases')

    logger.info('Successfully generated top fraud cases by device and browser data.')
    return jsonify(top_device_browser_data.to_dict(orient='records'))



if __name__ == '__main__':
    logger.info('Starting Flask application...')
    app.run(debug=True)
