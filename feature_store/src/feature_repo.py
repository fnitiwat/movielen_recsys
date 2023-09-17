import pandas as pd
import numpy as np
from feast import (
    FileSource,
    Entity,
    FeatureView,
    FeatureService,
    Field,
    types,
    ValueType,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Int64


################################
# rating
################################
# TODO: make rating correct (support 2 key userId, movieId)
rating_entity = Entity(name="rating_entity", join_keys=["userId"])
rating_feature_view = FeatureView(
    name="rating_feature_view",
    entities=[rating_entity],
    source=FileSource(
        name="rating_source",
        path="./data/interim/transformed_ratings.parquet",
        timestamp_field="timestamp",
    ),
    ttl=None,
    online=True,
    tags={},
)


################################
# watch histories
################################
watch_histories_entity = Entity(name="watch_histories_entity", join_keys=["userId"])
watch_histories_feature_view = FeatureView(
    name="watch_histories_feature_view",
    entities=[watch_histories_entity],
    source=FileSource(
        name="watch_histories_source",
        path="./data/interim/transformed_watch_histories.parquet",
        timestamp_field="timestamp",
    ),
    ttl=None,
    online=True,
    tags={},
)

################################
# all movies
################################
global_stats_view = FeatureView(
    name="global_stats_view",
    entities=[],
    source=FileSource(
        name="global_stats_source",
        path="./data/interim/transformed_global_stats.parquet",
        timestamp_field="timestamp",
    ),
)


# ################################
# # unwatch histories
# ################################
# # Use the input data and feature view features to create new features
# # Note: this function still bug cant't use now
# # TODO: fix this function
# @on_demand_feature_view(
#     sources=[watch_histories_feature_view, global_stats_view],
#     schema=[Field(name="unwatch_histories", dtype=types.String)],
# )
# def unwatch_histories(features_df: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame(columns=["unwatch_histories"])
#     convert_dict = {"unwatch_histories": "string"}
#     df = df.astype(convert_dict)
#     return df


################################
# Service
################################
rating_fs = FeatureService(name="rating_service", features=[rating_feature_view])
watch_histories_fs = FeatureService(
    name="watch_histories_service",
    features=[
        watch_histories_feature_view,
        # unwatch_histories,
    ],
)
model_v1_fs = FeatureService(name="model_v1", features=[rating_feature_view])
model_v2_fs = FeatureService(name="model_v2", features=[watch_histories_feature_view])
model_v3_fs = FeatureService(name="model_v3", features=[rating_feature_view, watch_histories_feature_view])
