import pytest
from typing import List
from modules.feature_store import FeatureStore


class MockDB:
    def __init__(self):
        self.user_ids = [1]

    def get_user_ids(self) -> List[int]:
        return self.user_ids

    def get_watched_movie_ids_from_user_id(self, user_id: int) -> List[int]:
        if user_id in self.user_ids:
            return [100, 200, 300, 400]
        else:
            return []

    def get_unwatched_movie_ids_from_user_id(self, user_id: int) -> List[int]:
        if user_id in self.user_ids:
            return [500]
        else:
            return [100, 200, 300, 400]


class TestFeatureStore:
    mock_db = MockDB()
    feature_store = FeatureStore(db=mock_db)

    def test_get_features_user_inside_db(self):
        features = self.feature_store.get_features(1)
        assert features == {
            "histories": [100, 200, 300, 400],
            "unwatched_movie_ids": [500],
        }

    def test_get_features_user_outside_db(self):
        features = self.feature_store.get_features(-100)
        assert features == None
