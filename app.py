import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from recommender.prediction import MovieRecommenderPrediction
from recommender.training import MovieRecommenderTraining

load_dotenv()

# Training phase
mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
training = MovieRecommenderTraining(mongodb_connection_string)
training.train("models/model.pkl")

# Loading model
prediction = MovieRecommenderPrediction("models/model.pkl")
prediction.load_model()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    liked_movie_ids = data.get('liked_movie_ids', [])
    movie_ids = data.get('movie_ids', [])

    predictions = []

    for movie_id in movie_ids:
        prediction_result = prediction.predict_user_preference_extended(
            liked_movie_ids, movie_id)
        predictions.append(
            {
                "movie_id": movie_id,
                "tfidf": prediction_result[0],
                "cosine": prediction_result[1],
            }
        )

    return jsonify(predictions)


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        training = MovieRecommenderTraining(mongodb_connection_string)
        training.train("models/model.pkl")

        global prediction
        prediction.load_model()

        return jsonify({"status": "success", "message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
