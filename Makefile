setup:
	pip3 install -r requirements.txt
	
serve:
	cd app/ && python3 -m uvicorn main:app --host=0.0.0.0 --port=8080 --reload

quality_checks:
	black .

test:
	cd app/ && pytest
	
docker_build:
	docker build -t movielen_recsys .

docker_run:
	docker run -p 8080:8080 movielen_recsys 