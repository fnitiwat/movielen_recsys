import surprise
import os
from fastapi import FastAPI
from typing import Union

from modules.db import DB
from modules.feature_store import FeatureStore
from modules.recommender import Recommender
from schema import RecommendResponse, RecommendWithMetadataResponse, FeaturesResponse


app = FastAPI()


class Config:
    base_path = os.path.abspath(os.path.dirname(__file__))
    links_csv_path = os.path.join(base_path, "./modules/artifacts/cleaned_links.csv")
    movies_csv_path = os.path.join(base_path, "./modules/artifacts/cleaned_movies.csv")
    ratings_csv_path = os.path.join(
        base_path, "./modules/artifacts/cleaned_ratings.csv"
    )
    tags_csv_path = os.path.join(base_path, "./modules/artifacts/cleaned_tags.csv")
    model_path = os.path.join(
        base_path, "./modules/artifacts/algorithm_instance.pickle"
    )
    artifact_dir = os.path.join(base_path, "./modules/artifacts")


config = Config()


db = DB(
    links_csv_path=config.links_csv_path,
    movies_csv_path=config.movies_csv_path,
    ratings_csv_path=config.ratings_csv_path,
    tags_csv_path=config.tags_csv_path,
)
feature_store = FeatureStore(db=db)

_, model = surprise.dump.load(config.model_path)
recommender = Recommender(model=model, db=db, feature_store=feature_store)


@app.get(
    path="/recommendations",
    response_model=Union[RecommendWithMetadataResponse, RecommendResponse],
)
async def get_recommend(user_id: int, returnMetadata: bool = False):
    if returnMetadata:
        recommend_dict = recommender.recommend_with_metadata(user_id)
    else:
        recommend_dict = recommender.recommend(user_id)
    return recommend_dict


@app.get(path="/features", response_model=FeaturesResponse)
async def get_features(user_id: int):
    features_dict = feature_store.get_features(user_id=user_id)
    features_dict = {"features": [features_dict]}
    return features_dict
