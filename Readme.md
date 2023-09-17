# Movielen (small) Recsys

### Built With

- FastAPI (serve)
- Surprise-ScikitLearn (model)
- pytest (test)

### How do I get set up?

- on local machine (train, serve, load test)
  ```
  make setup
  ```
- on docker
  ```
  make docker_build
  ```

### Usage

- train (on local machine)

  - run from train.py
    ```
    make train
    ```
  - (recommend) run from notebook as /research_phase/train.ipynb

- serve
  - on local machine
  ```
  make serve
  ```
  - on docker
  ```
  make docker_run_serve
  ```
  - after start serve (local or docker)
    - endpoint is http://localhost:8080/recommendations
    - api spec is on http://localhost:8080/redoc and http://localhost:8080/docs
- test
  ```
  make test_serve
  ```

### How does it work

- #### Model
  - user based collabortation filtering with SVD
- #### Serve Code Structure

  - main.py
    - this module is wrap recommender and serve as RestAPI by FastAPI
  - modules/db.py
    - this module is use for query movielen data
    - now it just load movielen data as dataframe
  - modules/feature_store.py
    - this module is use for query feature to use for model prediction
    - now return features are "histories" (watched movie id) and "unwatched_movie_ids"
    - now it just query from DB, but in future if model is more complex we can get benefit from this module
  - modules/recommender.py
    - this module is wrap model to use for recommend data from user_id
    - given user_id and use SVD model to predict rating of unwatch movies
    - then get top 10 movie and return to user

- #### Research Phase (train)
  - clean data
    - use pandasprofiling to see static of data
    - merge movie in movie_df that have duplicate title
    - remove rating data that have rating outside range[0, 5] and user_id is Nan
  - train model
    - user surprise lib to train SVD model
    - get RMSE of algorithm on train test data split
    - get Hitrate (precision) of this algorithm on leave one out data split
    - get final model by use all data as trainset (to memory it)
    - save artifacts

### How to feed input and get output

- by Swagger UI
  - after serve access this url http://localhost:8080/docs
  - you can use interaction web UI to try call API
- by API
  - you can call API follow specification by your prefer tools ex. curl, python requests, postman

### Input & Output Example

- see in folder images

