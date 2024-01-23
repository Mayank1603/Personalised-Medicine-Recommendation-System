#Search By Reason
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)
CORS(app) 
medicine = pd.read_csv('merged_medicine_data.csv')
def get_recommendations_from_model(user_condition):
    df = medicine[['Drug_Name', 'Description', 'Reason']].copy()
    df['Reason'] = df['Reason'].astype(str)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Reason'])
    user_condition_tfidf = tfidf_vectorizer.transform([user_condition])
    similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1][:10]
    top_medicines = df[['Drug_Name', 'Description']].iloc[top_indices].to_dict(orient='records')
    return top_medicines
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_condition = request.args.get('user_condition')
    if user_condition is not None:
        recommendations = get_recommendations_from_model(user_condition)
        return jsonify({"recommendations": recommendations})
    else:
        return jsonify({"error": "Missing 'user_condition' parameter"}), 400
if __name__ == '__main__':
    app.run(debug=True)