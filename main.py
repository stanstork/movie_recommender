from recommender.prediction import MovieRecommenderPrediction
from recommender.training import MovieRecommenderTraining

# Training phase
training = MovieRecommenderTraining()
training.train("models/model.pkl")

prediction = MovieRecommenderPrediction("models/model.pkl")
prediction.load_model()