from recommendations import get_recommendations
from flask import Flask, request, jsonify
from celery import celery

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def recommend():

    auth_header = request.headers.get('Authorization')

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    user_token = auth_header.split("Bearer ")[1]

    recommendations = get_recommendations(user_token)

    if "error" in recommendations:
        return jsonify(recommendations), 400

    return jsonify(recommendations)
    
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)