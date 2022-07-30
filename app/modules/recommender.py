from typing import Dict, List, Tuple
from collections import defaultdict

from .feature_store import FeatureStore
from .db import DB


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
