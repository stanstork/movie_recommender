import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from recommender.prediction import MovieRecommenderPrediction
from recommender.training import MovieRecommenderTraining

load_dotenv()

app = Flask(__name__)

# Training phase
mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
training = MovieRecommenderTraining(mongodb_connection_string)
training.train("models/model.pkl")

# Loading model
prediction = MovieRecommenderPrediction("models/model.pkl")
prediction.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    liked_movies = data.get('liked_movies', [])
    movie_id = data.get('movie_id', 0)
    
    # Call the prediction method
    predictions = prediction.predict_user_preference_extended(liked_movies, movie_id)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
