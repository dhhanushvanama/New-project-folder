from flask import Flask, request, render_template
import pandas as pd
import sys
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    try:
        data = CustomData(
            debt_ratio=float(request.form['debt_ratio']),
            total_debt_net_worth=float(request.form['total_debt_net_worth']),
            cash_total_assets=float(request.form['cash_total_assets']),
            net_worth_assets=float(request.form['net_worth_assets']),
            operating_profit_rate=float(request.form['operating_profit_rate']),
            interest_expense_ratio=float(request.form['interest_expense_ratio']),
        )

        input_df = data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        result = "Bankrupt" if prediction[0] == 1 else "Not Bankrupt"
        return render_template('index.html', results=result)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(debug=True)