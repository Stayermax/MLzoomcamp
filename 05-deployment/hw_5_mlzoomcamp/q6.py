import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('q4')


def load_model(model_file_path: str, dict_vectorizer_file_path: str) -> tuple([LogisticRegression, DictVectorizer]):
    with open(model_file_path, 'rb') as f_in: 
	    model = pickle.load(f_in)


    with open(dict_vectorizer_file_path, 'rb') as f_in: 
	    dv = pickle.load(f_in)

    return model, dv

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    model, dv = load_model('./model2.bin', './dv.bin')
    X = dv.transform([client])
    score = model.predict_proba(X)[0,1]

    result = {
        "churn_probability": float(score)
    }
    print(f"result: {result}")
    return jsonify(result)


if __name__ == "__main__":
    app.run(
		debug=True,
		host='0.0.0.0', 
		port=9696
	)
    

