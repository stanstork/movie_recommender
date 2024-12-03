import pickle
# from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class MovieRecommenderTraining:
    def __init__(self, mongodb_connection_string):
        self.movies_metadata = None
        # self.stemmer = SnowballStemmer("english")
        self.cosine_sim = None
        self.titles = None
        self.indices = None
        self.mongodb_connection_string = mongodb_connection_string

    def load_data(self):
        # Load data from MongoDB collection
        client = pymongo.MongoClient(self.mongodb_connection_string)
        db = client["movieDB"]
        collection = db["movieDetails"]
        self.movies_metadata = pd.DataFrame(list(collection.find()))

        print("Data loaded from MongoDB collection.")

    def preprocess_data(self):
        # Preprocess data
        print("Preprocessing data...")

        self.movies_metadata["genres"] = self.movies_metadata["genres"].apply(
            lambda x: [i["name"] for i in x]
        )
        self.movies_metadata["genres"] = self.movies_metadata["genres"].apply(
            lambda x: x * 2
        )
        self.movies_metadata["cast"] = self.movies_metadata["cast"].apply(
            lambda x: [i["name"] for i in x] if isinstance(x, list) else []
        )
        self.movies_metadata["cast"] = self.movies_metadata["cast"].apply(
            lambda x: x[:3] if len(x) >= 3 else x
        )
        self.movies_metadata["cast"] = self.movies_metadata["cast"].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )
        self.movies_metadata["director"] = self.movies_metadata["crew"].apply(
            lambda x: next((i["name"]
                           for i in x if i["job"] == "Director"), np.nan)
        )
        self.movies_metadata["director"] = self.movies_metadata["director"].astype("str").apply(
            lambda x: [str.lower(x.replace(" ", ""))] * 3
        )
        self.movies_metadata["keywords"] = self.movies_metadata["keywords"].apply(
            lambda x: [i["name"] for i in x] if isinstance(x, list) else []
        )
        # self.movies_metadata["keywords"] = self.movies_metadata["keywords"].apply(
        #     lambda x: [self.stemmer.stem(i) for i in x]
        # )
        self.movies_metadata["keywords"] = self.movies_metadata["keywords"].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )
        self.movies_metadata["soup"] = (
            self.movies_metadata["keywords"]
            + self.movies_metadata["cast"]
            + self.movies_metadata["director"]
            + self.movies_metadata["genres"]
        )
        self.movies_metadata["soup"] = self.movies_metadata["soup"].apply(
            lambda x: " ".join(x)
        )

        print("Data preprocessed.")

    def calculate_similarity(self):
        # Calculate similarity using CountVectorizer and cosine similarity
        print("Calculating similarity...")

        count = CountVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=0.0, stop_words="english"
        )
        count_matrix = count.fit_transform(self.movies_metadata["soup"])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.movies_metadata = self.movies_metadata.reset_index()
        self.titles = self.movies_metadata["title"]
        self.indices = pd.Series(
            self.movies_metadata.index, index=self.movies_metadata["title"]
        )

        print("Similarity calculated.")

    def save_model(self, file_path):
        # Save the trained model to a file
        model = {
            "movies_metadata": self.movies_metadata,
            "cosine_sim": self.cosine_sim,
            "titles": self.titles,
            "indices": self.indices
        }
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        print("Model saved to:", file_path)

    def train(self, file_path):
        # Train the movie recommender system
        print("Training movie recommender system...")

        self.load_data()
        self.preprocess_data()
        self.calculate_similarity()
        self.save_model(file_path)

        print("Movie recommender system training completed.")
