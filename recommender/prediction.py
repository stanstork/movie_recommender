import pickle
import numpy as np

class MovieRecommenderPrediction:
    def __init__(self, model_file):
        self.model_file = model_file
        self.movies_metadata = None
        self.cosine_sim = None
        self.titles = None
        self.indices = None

    def load_model(self):
        # Load the trained model from a file
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        self.movies_metadata = model["movies_metadata"]
        self.cosine_sim = model["cosine_sim"]
        self.titles = model["titles"]
        self.indices = model["indices"]

        print("Model loaded from:", self.model_file)

    def predict_user_preference(self, liked_movies, movie_id):
        # Predict user preference based on cosine similarity
        liked_movie_indices = self.movies_metadata.index[
            self.movies_metadata["_id"].isin(liked_movies)
        ]
        target_index = self.movies_metadata.index[
            self.movies_metadata["_id"] == movie_id
        ][0]

        similarity_scores = self.cosine_sim[target_index][liked_movie_indices]
        avg_similarity_score = np.mean(similarity_scores)

        print("User preference prediction (Cosine Similarity):",
              avg_similarity_score)

        return avg_similarity_score