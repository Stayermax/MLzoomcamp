import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

def load_model(model_file_path: str, dict_vectorizer_file_path: str) -> tuple([LogisticRegression, DictVectorizer]):
    with open(model_file_path, 'rb') as f_in: 
	    model = pickle.load(f_in)


    with open(dict_vectorizer_file_path, 'rb') as f_in: 
	    dv = pickle.load(f_in)

    return model, dv

def q3():
    model, dv = load_model('./model1.bin', './dv.bin')
    client = {"job": "management", "duration": 400, "poutcome": "success"}
    X = dv.transform([client])
    score = model.predict_proba(X)[0,1]
    print(f"Q3: probability: {score}")

if __name__ == "__main__":
    q3()
    

