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

  ```
  make train
  ```

- serve
  - on local machine
  ```
  make serve
  ```
  - on docker
  ```
  make docker_run
  ```
  - after start serve (local or docker)
    - endpoint is http://localhost:808/recommend 
    - api spec is on http://localhost:8089/redoc and http://localhost:8089/docs
- load test (on local machine)
  ```
  make loadtest
  ```
  - before start load test you have to start serve (local or docker)
  - after spinup load test you have to access http://localhost:8089 to run load test and see TPS, latency result


### Model Idea
 - use collaborate filtering by KNN to find top 10 lowest cosine similarity
 - unseen user will get recommend with top 10 highest frequency restaurants

### Assignment answer
 - in docs/Answer MLE Test.docx.pdf

### Who do I talk to ?

- fnitiwat001@gmail.com
