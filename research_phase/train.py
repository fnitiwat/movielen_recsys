
def display(x):
    print(x)
    
# %%
import surprise
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm.notebook import tqdm
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from pandas_profiling import ProfileReport
from collections import defaultdict


# %% [markdown]
# # Config
# 

# %%
class Config:
    links_csv_path = "../data/movielens-small/links.csv"
    movies_csv_path = "../data/movielens-small/movies.csv"
    ratings_csv_path = "../data/movielens-small/ratings.csv"
    tags_csv_path = "../data/movielens-small/tags.csv"
    artifact_dir = "./artifacts/"


config = Config()


# %% [markdown]
# # Load data
# 

# %%
links_df = pd.read_csv(config.links_csv_path)
movies_df = pd.read_csv(config.movies_csv_path)
ratings_df = pd.read_csv(config.ratings_csv_path)
tags_df = pd.read_csv(config.tags_csv_path)


# %% [markdown]
# # Clean Data
# 

# %%
# run pandas profile to see stat of data

# for curr_df in [links_df, movies_df, ratings_df, tags_df]:
#     profile = ProfileReport(curr_df, title="Pandas Profiling Report")
#     profile.to_widgets()


# %%
# Movie: Title is duplicate

# get movie that have duplicate title
duplicate_title_movie_df = movies_df[
    movies_df.title.isin(movies_df[movies_df.duplicated("title")].title.tolist())
]
print("duplicate tile movies")
display(duplicate_title_movie_df)


# %%
# remove duplicate movie
movies_df = movies_df[~movies_df.duplicated("title")]

print("movie after remove duplicate")
display(movies_df)


# TODO: instead just remove -> merge genres (if u want to use genres to do something)
# TODO: create map movieId and convert movieId to all data
def get_duplicate_movie_id_mapping() -> Dict[str, str]:
    """get dictionary that map movie_id that have duplicate title to another id that is we want to use
    Returns:
        Dict[str, str]: dictionary that map input_movie_id to target_movie_id
    """


# %%
# Links: Found  missing in tmdlbld (but not do anything)

not_in_tmdbld_movie_ids = links_df[links_df.tmdbId.isna()].movieId.tolist()
# show row that tmfbId is nan
display(movies_df[movies_df.movieId.isin(not_in_tmdbld_movie_ids)])

# i think this happen because movie have in imdbld but not in tmdbld
# should not have effect much


# %%
# Ratings: Remove row rating outside rang[0,5] and remove row userId missing

# show row where rating value outside range
print("row where rating outside range:")
display(ratings_df[((ratings_df.rating > 5) | (ratings_df.rating < 0))])
print("")

# show row where user is nan
print("row where user is nan:")
display(ratings_df[ratings_df.userId.isna()])
print("")

# remove row where rating is outside [0,5]
ratings_df = ratings_df[~((ratings_df.rating > 5) | (ratings_df.rating < 0))]

# remove row where user is nan
ratings_df = ratings_df[~ratings_df.userId.isna()]

# check after remove row that have rating out side range [0, 5]
print("unique rating value after clean:")
print(np.unique(ratings_df.rating, return_counts=True))
print("")

# check to confirm not have nan user
print("check is nan in df after clean:")
display(ratings_df[ratings_df.userId.isna()])


# %%
# reset index
ratings_df = ratings_df.reset_index(drop=True)
ratings_df = ratings_df.astype({"userId": "int", "movieId": "int"})
print("ratings")
display(ratings_df)

# set movieId as index
movies_df = movies_df.set_index("movieId")
# display(movies_df)

# change column name
movies_df.columns = movies_df.columns.str.replace("title", "movie_name")
movies_df.columns = movies_df.columns.str.replace("genres", "genre")
print("movies")
display(movies_df)


# %%
# save cleaned data
os.makedirs(config.artifact_dir, exist_ok=True)
cleaned_movies_csv_path = os.path.join(config.artifact_dir, "cleaned_movies.csv")
cleaned_rating_csv_path = os.path.join(config.artifact_dir, "cleaned_ratings.csv")
cleaned_links_csv_path = os.path.join(config.artifact_dir, "cleaned_links.csv")
cleaned_tags_csv_path = os.path.join(config.artifact_dir, "cleaned_tags.csv")

movies_df.to_csv(cleaned_movies_csv_path)
ratings_df.to_csv(cleaned_rating_csv_path)
links_df.to_csv(cleaned_links_csv_path)
tags_df.to_csv(cleaned_tags_csv_path)

del movies_df
del ratings_df
del links_df
del tags_df


# %% [markdown]
# # Trainer
# 

# %%
class Trainer:
    def __init__(self, ratings_df, config: Config):
        self.config = config
        self.ratings_df = ratings_df

    def run(self):
        self._get_hitrate_results()
        print("--------------------------------------")
        self._get_rmse_result()
        print("--------------------------------------")

        # train full dataset
        # init data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[["userId", "movieId", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        # init model
        algorithm_instance = self._init_model()

        # train
        algorithm_instance.fit(trainset)

        # save artifacts
        self._save_artifacts(
            algorithm_instance,
            predictions=None,
            trainset=trainset,
            artifact_dir=config.artifact_dir,
        )

    def _init_model(self) -> surprise.prediction_algorithms.algo_base.AlgoBase:
        print("init model")

        algorithm_instance = SVD(biased=False)
        return algorithm_instance

    def _save_artifacts(
        self, algorithm_instance, predictions, trainset, artifact_dir: str
    ) -> None:
        print("save artifacts")

        algorithm_instance_save_path = os.path.join(
            artifact_dir, "algorithm_instance.pickle"
        )
        os.makedirs(os.path.dirname(algorithm_instance_save_path), exist_ok=True)
        surprise.dump.dump(
            algorithm_instance_save_path,
            algo=algorithm_instance,
            predictions=predictions,
        )

        trainset_path = os.path.join(artifact_dir, "surprise_trainset.pickle")

        with open(trainset_path, "wb") as f:
            pickle.dump(trainset, f)

    def _get_hitrate_results(self):
        """evaluate hit rate but not use model just to check how well of this algorithm"""

        def _get_top_N(predictions, n=10, minimumRating=4.0):
            topN = defaultdict(list)

            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                if estimatedRating >= minimumRating:
                    topN[userID].append((movieID, estimatedRating))

            for userID, ratings in topN.items():
                ratings.sort(key=lambda x: x[1], reverse=True)
                topN[userID] = ratings[:n]

            return topN

        def _HitRate(topNPredicted, leftOutPredictions):
            hits = 0
            total = 0

            # For each left-out rating
            for leftOut in leftOutPredictions:
                userID = leftOut[0]
                leftOutMovieID = leftOut[1]
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[userID]:
                    if leftOutMovieID == movieID:
                        hit = True
                        break
                if hit:
                    hits += 1

                total += 1

            # Compute overall precision
            return hits / total

        print("evaluate hit rate")

        # init data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[["userId", "movieId", "rating"]], reader
        )

        # split train, test leave one out for eval hit rate
        splitter = LeaveOneOut(n_splits=1, random_state=1)
        train_loocv, test_loocv = list(splitter.split(data))[0]

        algorithm_instance = self._init_model()
        algorithm_instance.fit(train_loocv)
        left_out_predictions = algorithm_instance.test(test_loocv)
        loocv_anti_testset = train_loocv.build_anti_testset()
        all_predictions = algorithm_instance.test(loocv_anti_testset)
        top_n_predicted = _get_top_N(all_predictions)
        hitrate = _HitRate(top_n_predicted, left_out_predictions)
        print(f"HitRate: {hitrate}")

        return all_predictions

    def _get_rmse_result(self):
        """evaluate rmse but not use model just to check how well of this algorithm"""

        def _evaluate_RMSE(
            algorithm_instance: surprise.prediction_algorithms.algo_base.AlgoBase,
            testset,
        ) -> any:
            predictions = [
                algorithm_instance.predict(uid, iid, r_ui_trans, verbose=False)
                for (uid, iid, r_ui_trans) in tqdm(testset, desc="making predictions")
            ]
            eval_report = accuracy.rmse(predictions)
            return eval_report, predictions

        print("evaluate rmse")

        # init data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[["userId", "movieId", "rating"]], reader
        )
        # split train, test for eval RMSE
        trainset, testset = train_test_split(data, test_size=0.25)

        # init model
        algorithm_instance = self._init_model()

        # train
        algorithm_instance.fit(trainset)

        _evaluate_RMSE(algorithm_instance=algorithm_instance, testset=testset)


ratings_df = pd.read_csv(cleaned_rating_csv_path)
trainer = Trainer(ratings_df=ratings_df, config=config)
trainer.run()


# %% [markdown]
# # Inference
# 

# %%
class DB:
    def __init__(
        self,
        links_csv_path: str,
        movies_csv_path: str,
        ratings_csv_path: str,
        tags_csv_path: str,
    ):
        self.links_df = pd.read_csv(links_csv_path)
        self.movies_df = pd.read_csv(movies_csv_path).set_index("movieId")
        self.ratings_df = pd.read_csv(ratings_csv_path)
        self.tags_df = pd.read_csv(tags_csv_path)

    def get_metadata_from_movie_id(self, movie_id: int) -> Dict:
        selected_row = self.movies_df.loc[movie_id]
        metadata = {
            "id": str(movie_id),
            "title": selected_row.movie_name,
            "genres": selected_row.genre.split("|"),
        }
        return metadata

    def get_watched_movie_ids_from_user_id(self, user_id: int) -> List[int]:
        watched_movie_ids = self.ratings_df[
            self.ratings_df.userId == user_id
        ].movieId.tolist()
        return watched_movie_ids

    def get_unwatched_movie_ids_from_user_id(self, user_id: int) -> List[int]:
        watched_movie_ids = self.get_watched_movie_ids_from_user_id(user_id)
        unwatched_movie_ids = list(
            set(self.movies_df.index.tolist()) - set(watched_movie_ids)
        )
        return unwatched_movie_ids

    def get_user_ids(self) -> List[int]:
        return list(set(self.ratings_df.userId))

    def get_top_k_popular_movie_ids(self, k: int) -> List[int]:
        top_k_popular_movie_ids = (
            self.ratings_df.groupby("movieId")
            .sum()
            .sort_values("rating", ascending=False)
            .iloc[:k]
            .index.tolist()
        )
        top_k_popular_movie_ids = [
            int(movie_id) for movie_id in top_k_popular_movie_ids
        ]
        return top_k_popular_movie_ids


# %%
class FeatureStore:
    def __init__(self, db: DB):
        self.db = db
        self.user_ids = self.db.get_user_ids()

    def get_features(self, user_id: int) -> Dict:
        # if user not in user_id db then return None
        if user_id not in self.user_ids:
            return None

        feature_dict = {
            "histories": self.db.get_watched_movie_ids_from_user_id(user_id=user_id),
            "unwatched_movie_ids": self.db.get_unwatched_movie_ids_from_user_id(
                user_id=user_id
            ),
        }

        return feature_dict


# %%
class Recommender:
    def __init__(self, model, feature_store: FeatureStore, db: DB):
        self.model = model
        self.feature_store = feature_store
        self.db = db
        self.k = 10  # n recommend movies to return

    def recommend(self, user_id: int) -> Dict:
        user_ids = self.db.get_user_ids()
        if user_id not in user_ids:
            top_k_movie_ids = self._get_popular_movie_ids()

        else:
            features = self.feature_store.get_features(user_id)

            predictions = []
            for unwatched_movie_id in self._remove_unrated_movie_ids(
                features["unwatched_movie_ids"]
            ):

                prediction = self.model.predict(user_id, unwatched_movie_id)

                if prediction.details["was_impossible"] == True:
                    inner_uid = user_id
                    inner_iid = unwatched_movie_id
                    raise Exception(
                        f"bug: user_id: {user_id} item_id: {unwatched_movie_id} inner_uid: {inner_uid} inner_iid: {inner_iid} predictions: {prediction}"
                    )
                else:
                    # print(user_id, unwatched_movie_id)
                    pass

                predictions.append(prediction)

            top_k_predictions = self._get_top_k(predictions, k=self.k)
            top_k_movie_ids = [movie_id for movie_id, rating in top_k_predictions]

        ouput_dict = {"items": [{"id": str(movie_id)} for movie_id in top_k_movie_ids]}

        return ouput_dict

    def recommend_with_metadata(self, user_id: int) -> Dict:
        items_dict = self.recommend(user_id)
        for i, item_data in enumerate(items_dict["items"]):
            movie_id = int(item_data["id"])
            metadata = self.db.get_metadata_from_movie_id(movie_id)
            items_dict["items"][i].update(metadata)

        return items_dict

    def _get_popular_movie_ids(self):
        return self.db.get_top_k_popular_movie_ids(k=self.k)

    def _get_top_k(self, predictions, k=10) -> List[Tuple[int, float]]:
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            topN[userID].append((movieID, estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[userID] = ratings[:k]

        return topN[userID]

    def _remove_unrated_movie_ids(self, movie_ids: List[int]) -> List[str]:
        """because have some movie that haven't rate that make SVD model dont have information about it
        so this function is harcode remove unrated_movie_ids that prevent model error
        # TODO: improve this process
        """
        unrated_movie_ids = [
            298,
            1076,
            1574,
            2824,
            2939,
            2964,
            3192,
            3338,
            3456,
            3914,
            4116,
            4194,
            4384,
            5241,
            5272,
            5721,
            5723,
            5745,
            5746,
            5764,
            5884,
            6668,
            6835,
            6849,
            7020,
            7792,
            7899,
            8765,
            25855,
            26085,
            30892,
            32160,
            32371,
            34482,
            55391,
            66320,
            85565,
            103606,
            110718,
            112868,
            114184,
            127184,
            128488,
            130578,
            131023,
            165551,
        ]
        movie_ids = list(set(movie_ids) - set(unrated_movie_ids))
        return movie_ids


# %%
# test recommender
model_path = os.path.join(config.artifact_dir, "algorithm_instance.pickle")
_, model = surprise.dump.load(model_path)

db = DB(
    links_csv_path=os.path.join(config.artifact_dir, "cleaned_links.csv"),
    movies_csv_path=os.path.join(config.artifact_dir, "cleaned_movies.csv"),
    ratings_csv_path=os.path.join(config.artifact_dir, "cleaned_ratings.csv"),
    tags_csv_path=os.path.join(config.artifact_dir, "cleaned_tags.csv"),
)
feature_store = FeatureStore(db=db)
recommender = Recommender(model=model, feature_store=feature_store, db=db)


# %%
# test can recommend all user
popular_output = recommender.recommend(-100)

user_ids = db.get_user_ids()
for user_id in user_ids:
    # print("user_id:", user_id)
    output = recommender.recommend(user_id)
    assert output != popular_output


# %% [markdown]
# # Push Artifacts
# 

# %%
# ! cp -r $config.artifact_dir ../app/modules/artifacts



