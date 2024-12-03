import os
import threading

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from recommender.prediction import MovieRecommenderPrediction
from recommender.training import MovieRecommenderTraining

load_dotenv()

# Training phase
mongodb_connection_string = "mongodb://user:pass@localhost:27017"#os.getenv("MONGODB_CONNECTION_STRING")
training = MovieRecommenderTraining(mongodb_connection_string)
training.train("models/model.pkl")

# Loading model
prediction = MovieRecommenderPrediction("models/model.pkl")
prediction.load_model()

app = Flask(__name__)


def retrain_model():
    try:
        mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        training = MovieRecommenderTraining(mongodb_connection_string)
        training.train("models/model.pkl")

        global prediction
        prediction.load_model()

        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    liked_movie_ids = data.get('liked_movie_ids', [])
    movie_ids = data.get('movie_ids', [])

    predictions = []

    for movie_id in movie_ids:
        avg_similarity_score = prediction.predict_user_preference(liked_movie_ids, movie_id)
        predictions.append(
            {
                "movie_id": movie_id,
                "score": avg_similarity_score,
            }
        )

    return jsonify({
        "predictions": predictions
    })


@app.route('/retrain', methods=['POST'])
def retrain():
    threading.Thread(target=retrain_model).start()
    return jsonify({"status": "success", "message": "Model retraining started"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
