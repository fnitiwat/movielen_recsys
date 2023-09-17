"""
transform raw data to use as feature store data source
"""
from datetime import datetime
import pandas as pd


def transform_ratings_df(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = df.timestamp.apply(lambda x: datetime.fromtimestamp(x))
    df["userId"] = df["userId"].apply(lambda x: str(x))
    return df

def transform_ratings_to_last_watch_histories(df: pd.DataFrame, n_last_watch_history: int = 100) -> pd.DataFrame:
    def _agg_last_watch_histories(group):
        x = [window.to_list()[::-1] for window in group['movieId'].rolling(n_last_watch_history)]
        group["watch_histories"] = x
        return group

    df = df.groupby('userId', group_keys=True).apply(_agg_last_watch_histories).reset_index(level=0, drop=True)
    df = df[["userId", "watch_histories", "timestamp"]]
    return df

def get_global_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    movie_ids = sorted(df["movieId"].unique())
    output_df = pd.DataFrame({
        "movieIds": [movie_ids],
        "timestamp": [datetime.now()]
    })
    return output_df




if __name__ == "__main__":
    src_path: str = "./data/raw/ratings.csv"
    src_movie_path: str = "./data/raw/movies.csv"
    des_rating_csv_path: str = "./data/interim/transformed_ratings.csv"
    des_rating_parquet_path: str = "./data/interim/transformed_ratings.parquet"
    des_watch_histories_csv_path: str = "./data/interim/transformed_watch_histories.csv"
    des_watch_histories_parquet_path: str = "./data/interim/transformed_watch_histories.parquet"
    des_global_stats_csv_path: str = "./data/interim/transformed_global_stats.csv"
    des_global_stats_parquet_path: str = "./data/interim/transformed_global_stats.parquet"
    rating_df = pd.read_csv(src_path)

    transformed_rating_df = transform_ratings_df(rating_df)
    transformed_rating_df.to_csv(des_rating_csv_path)
    transformed_rating_df.to_parquet(des_rating_parquet_path)

    watch_histories_df = transform_ratings_to_last_watch_histories(transformed_rating_df)
    watch_histories_df.to_csv(des_watch_histories_csv_path)
    watch_histories_df.to_parquet(des_watch_histories_parquet_path)

    movie_df = pd.read_csv(src_movie_path)
    global_stats_df = get_global_stats_df(movie_df)
    global_stats_df.to_csv(des_global_stats_csv_path)
    global_stats_df.to_parquet(des_global_stats_parquet_path)