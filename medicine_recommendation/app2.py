#Search By Reason, Symptoms and Medicine
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd
from prettytable import PrettyTable
app = Flask(__name__)
CORS(app)  
m_data = pd.read_csv('medical_data.csv')
def get_recommendations_from_model(user_condition, choice):
    df = m_data[['Causes', 'Symptoms', 'Disease', 'Medicine']].copy()
    df = df.drop_duplicates()
    tfidf_vectorizer = TfidfVectorizer()
    if choice == 'Medicine':
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Medicine'])
    elif choice == 'Symptoms':
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Symptoms'])
    else:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Disease'])
    user_condition_tfidf = tfidf_vectorizer.transform([user_condition])
    similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)
    threshold = 0.2  
    top_indices = [i for i, score in enumerate(similarity_scores[0]) if score > threshold]
    if top_indices:
        top_medicines = df[['Causes', 'Symptoms', 'Disease', 'Medicine']].iloc[top_indices].values.tolist()
        table = PrettyTable(['Causes', 'Symptoms', 'Disease', 'Medicine'])
        for med in top_medicines:
            table.add_row([med[0], med[1], med[2], med[3]])
        recommendations = table.get_html_string()
    else:
        recommendations = f"No matching medicines found for {user_condition}"
    return recommendations
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_condition = request.args.get('user_condition')
    choice = request.args.get('choice')
    if user_condition is not None and choice is not None:
        recommendations = get_recommendations_from_model(user_condition, choice)
        return recommendations
    else:
        return jsonify({"error": "Missing 'user_condition' or 'choice' parameter"}), 400
if __name__ == '__main__':
    host = '127.0.0.1'  
    port = 5501
    app.run(debug=True, host=host, port=port)