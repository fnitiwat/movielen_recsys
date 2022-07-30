from pydantic import BaseModel
from typing import List, Union


class Item(BaseModel):
    id: str


class ItemWithMetadata(BaseModel):
    id: str
    title: str
    genres: List[str]


class RecommendResponse(BaseModel):
    items: List[Item]


class RecommendWithMetadataResponse(BaseModel):
    items: List[ItemWithMetadata]


class Feature(BaseModel):
    histories: List[str]
    unwatched_movie_ids: List[str]


class FeaturesResponse(BaseModel):
    features: List[Union[Feature, None]]
