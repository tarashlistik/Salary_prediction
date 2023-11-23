from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)

one_hot_columns = joblib.load('/Users/admin/Downloads/Salary_prediction/save_model/one_hot_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    bins = [0, 1, 4, 7, 10, 15, float('inf')]
    labels = ['0', '1-3', '4-6', '7-10', '10-15', '15+']

    experience = float(request.form['experience'])
    position = request.form['position']

    input_data = pd.DataFrame([[experience, position]],
                              columns=['experience', 'position'])

    input_data['experience'] = pd.cut(input_data['experience'], bins=bins, labels=labels, right=False)

    input_data_encoded = pd.get_dummies(input_data, columns=['experience', 'position'])
    input_data_encoded = input_data_encoded.replace({False: 0, True: 1})

    missing_columns = set(one_hot_columns) - set(input_data_encoded.columns)
    for col in missing_columns:
        input_data_encoded[col] = 0

    input_data_encoded = input_data_encoded[one_hot_columns]


    model = joblib.load('/Users/admin/Downloads/Salary_prediction/save_model/decision_tree_model.pkl')
    # scaler = joblib.load('scaler.pkl')
    # input_data_encoded = scaler.fit_transform(input_data_encoded)

    salary_prediction = model.predict(input_data_encoded)

    rounded_prediction = round(salary_prediction[0] / 1000) * 1000

    return jsonify({'prediction': rounded_prediction})


if __name__ == '__main__':
    app.run(debug=True)