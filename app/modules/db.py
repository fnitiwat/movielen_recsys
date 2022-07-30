import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


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
