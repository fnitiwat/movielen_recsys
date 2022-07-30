import pytest
from modules.db import DB


class TestDB:
    db = DB(
        links_csv_path="./modules/artifacts/cleaned_links.csv",
        movies_csv_path="./modules/artifacts/cleaned_movies.csv",
        ratings_csv_path="./modules/artifacts/cleaned_ratings.csv",
        tags_csv_path="./modules/artifacts/cleaned_tags.csv",
    )

    def test_get_metadata_from_movie_df_user_inside_db(self):
        metadata = self.db.get_metadata_from_movie_id(1)
        assert isinstance(metadata, dict)
        assert isinstance(metadata["id"], str)
        assert isinstance(metadata["title"], str)
        assert isinstance(metadata["genres"], list)

    def test_get_metadata_from_movie_df_user_outside_db(self):
        with pytest.raises(KeyError) as e:
            self.db.get_metadata_from_movie_id(-100)

    def test_watched_movie_ids_from_user_id_user_inside_db(self):
        watched_movie_ids = self.db.get_watched_movie_ids_from_user_id(1)
        assert isinstance(watched_movie_ids, list)
        assert isinstance(watched_movie_ids[0], int)

    def test_watched_movie_ids_from_user_id_user_outside_db(self):
        watched_movie_ids = self.db.get_watched_movie_ids_from_user_id(-100)
        assert isinstance(watched_movie_ids, list)
        assert len(watched_movie_ids) == 0

    def test_unwatched_movie_ids_from_user_id_user_inside_db(self):
        unwatch_movie_ids = self.db.get_unwatched_movie_ids_from_user_id(1)
        assert isinstance(unwatch_movie_ids, list)
        assert isinstance(unwatch_movie_ids[0], int)

    def test_unwatched_movie_ids_from_user_id_user_outside_db(self):
        unwatch_movie_ids = self.db.get_unwatched_movie_ids_from_user_id(-100)
        assert isinstance(unwatch_movie_ids, list)
        assert isinstance(unwatch_movie_ids[0], int)
        assert len(unwatch_movie_ids) == 9737

    def test_get_user_ids(self):
        user_ids = self.db.get_user_ids()
        assert isinstance(user_ids, list)
        assert isinstance(user_ids[0], int)
        assert len(user_ids) == 590

    def test_get_top_k_popular_movie_ids(self):
        top_k_movie_ids = self.db.get_top_k_popular_movie_ids(k=10)
        assert isinstance(top_k_movie_ids, list)
        assert isinstance(top_k_movie_ids[0], int)
        assert len(top_k_movie_ids) == 10
