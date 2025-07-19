from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
import pandas as pd
import sys

sys.path.append('src')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect 6 inputs from form
        data = {
            'Industrial Risk': float(request.form['Industrial_Risk']),
            'Management Risk': float(request.form['Management_Risk']),
            'Financial Flexibility': float(request.form['Financial_Flexibility']),
            'Credibility': float(request.form['Credibility']),
            'Competitiveness': float(request.form['Competitiveness']),
            'Operating Risk': float(request.form['Operating_Risk'])
        }

        # Create DataFrame
        custom_data = CustomData(**data)
        input_df = custom_data.get_data_as_data_frame()

        # Predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        result = "Bankrupt" if prediction[0] == 1 else "Not Bankrupt"
        return render_template('home.html', results=result)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # ✅ Enable debug mode, but disable auto-reloader to preserve console output
    app.run(debug=True, use_reloader=False)
