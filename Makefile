setup:
	pip3 install -r requirements.txt
	
serve:
	cd app/ && python3 -m uvicorn main:app --host=0.0.0.0 --port=8080 --reload

train:
	cd research_phase && python3 train.py

quality_checks:
	black .

test_serve:
	cd app/ && pytest

docker_build:
	docker build -t movielen_recsys .

docker_run_serve:
	docker run -p 8080:8080 movielen_recsys 

docker_run_test_serve:
	docker run movielen_recsys make test_serve

docker_run_train:
	docker run movielen_recsys make train