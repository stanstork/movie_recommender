import pickle
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommenderPrediction:
    def __init__(self, model_file):
        self.model_file = model_file
        self.movies_metadata = None
        self.cosine_sim = None
        self.tfidf_matrix = None
        self.titles = None
        self.indices = None
    
    def load_model(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        self.movies_metadata = model["movies_metadata"]
        self.cosine_sim = model["cosine_sim"]
        self.tfidf_matrix = model["tfidf_matrix"]
        self.titles = model["titles"]
        self.indices = model["indices"]
        
    def predict_user_preference(self, liked_movies, movie_id):
        liked_movie_indices = self.movies_metadata.index[
            self.movies_metadata["id"].isin(liked_movies)
        ]
        target_index = self.movies_metadata.index[
            self.movies_metadata["id"] == movie_id
        ][0]

        similarity_scores = self.cosine_sim[target_index][liked_movie_indices]
        avg_similarity_score = np.mean(similarity_scores)

        return avg_similarity_score
    
    def predict_user_preference_tfidf(self, liked_movies, movie_id):
        liked_movie_indices = self.movies_metadata.index[
            self.movies_metadata["id"].isin(liked_movies)
        ]
        target_index = self.movies_metadata.index[
            self.movies_metadata["id"] == movie_id
        ][0]

        similarity_scores = linear_kernel(
            self.tfidf_matrix[target_index], self.tfidf_matrix[liked_movie_indices]
        ).flatten()
        avg_similarity_score = np.mean(similarity_scores)

        return avg_similarity_score

    def predict_user_preference_extended(self, liked_movies, movie_id):
        movie_ids = []
        for lm in liked_movies:
            movie_ids.append(lm)
            similar = self.get_recommendations(lm)
            for s in similar.values:
                movie_ids.append(s[0])

        tfidf_preference = self.predict_user_preference_tfidf(movie_ids, movie_id)
        cosine_preference = self.predict_user_preference(movie_ids, movie_id)
        
        return (tfidf_preference, cosine_preference)
    
    def get_recommendations(self, movie_id):
        idx = self.movies_metadata.index[self.movies_metadata["id"] == movie_id][0]

        # Compute the cosine similarity scores between the target movie and all other movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        # Retrieve the details of the recommended movies
        recommended_movies = self.movies_metadata.iloc[movie_indices][
            ["id", "title", "vote_count", "vote_average"]
        ]

        # Apply vote count and vote average thresholds to filter qualified movies
        vote_counts = recommended_movies[recommended_movies["vote_count"].notnull()][
            "vote_count"
        ].astype(float)
        vote_averages = recommended_movies[
            recommended_movies["vote_average"].notnull()
        ]["vote_average"].astype(float)
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)

        # Calculate the weighted rating for each qualified movie
        qualified = recommended_movies[
            (recommended_movies["vote_count"] >= m)
            & (recommended_movies["vote_count"].notnull())
            & (recommended_movies["vote_average"].notnull())
        ]
        qualified["vote_count"] = qualified["vote_count"].astype(float)
        qualified["vote_average"] = qualified["vote_average"].astype(float)
        qualified["wr"] = qualified.apply(
            lambda x: self.weighted_rating(x, m, C), axis=1
        )

        # Sort the movies based on the weighted rating and return the top 10 recommendations
        qualified = qualified.sort_values("wr", ascending=False).head(10)
        return qualified
    
    def weighted_rating(self, x, m, C):
        v = x["vote_count"]
        R = x["vote_average"]
        return (v / (v + m) * R) + (m / (m + v) * C)