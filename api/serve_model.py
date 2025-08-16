from flask import Flask, request, jsonify

import joblib


from logger import SetupLogger
# Initialize Flask app
app = Flask(__name__)

# Ensure logs directory exists
logger = SetupLogger(log_file='logs/app.log').get_logger()


# Load the trained model
model = joblib.load('gradient_boosting_fraud_best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud based on the input data
    """
    data = request.get_json(force=True)
    logger.info("Incoming request: %s", data)

    # Perform prediction
    try:
        prediction = model.predict(data['features'])
        logger.info("Prediction result: %s", prediction)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
