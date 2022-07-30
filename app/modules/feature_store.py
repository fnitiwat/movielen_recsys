from typing import Dict

from .db import DB


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
