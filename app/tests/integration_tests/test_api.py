from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_get_recommends_user_inside_db():
    user_id = "1"
    response = client.get(f"/recommendations?user_id={user_id}")
    response.raise_for_status()
    response_dict = response.json()

    assert isinstance(response_dict["items"], list)
    assert len(response_dict["items"]) == 10


def test_get_recommends_user_outside_db():
    user_id = "-100"
    response = client.get(f"/recommendations?user_id={user_id}")
    response.raise_for_status()
    response_dict = response.json()

    assert isinstance(response_dict["items"], list)
    assert len(response_dict["items"]) == 10


def test_get_recommends_with_metadata_user_inside_db():
    user_id = "1"
    response = client.get(f"/recommendations?user_id={user_id}&&returnMetadata=true")
    response.raise_for_status()
    response_dict = response.json()

    assert isinstance(response_dict["items"], list)
    assert len(response_dict["items"]) == 10


def test_get_recommends_with_metadata_user_outside_db():
    user_id = "-100"
    response = client.get(f"/recommendations?user_id={user_id}&&returnMetadata=true")
    response.raise_for_status()
    response_dict = response.json()

    assert isinstance(response_dict["items"], list)
    assert len(response_dict["items"]) == 10


def test_unsuccess_get_recommends_with_not_int():
    user_id = "x"
    response = client.get(f"/recommendations?user_id={user_id}")
    assert response.status_code == 422


def test_get_features_user_inside_db():
    user_id = "1"
    response = client.get(f"/features?user_id={user_id}")
    response.raise_for_status()
    response_dict = response.json()

    assert isinstance(response_dict["features"], list)
    assert len(response_dict["features"]) == 1

    assert isinstance(response_dict["features"][0]["histories"], list)
    assert isinstance(response_dict["features"][0]["histories"][0], str)

    assert isinstance(response_dict["features"][0]["unwatched_movie_ids"], list)
    assert isinstance(response_dict["features"][0]["unwatched_movie_ids"][0], str)


def test_get_features_user_outside_db():
    user_id = "-100"
    response = client.get(f"/features?user_id={user_id}")
    response.raise_for_status()
    response_dict = response.json()
    print(response_dict)

    assert isinstance(response_dict["features"], list)
    assert len(response_dict["features"]) == 1

    assert response_dict["features"][0] == None
