install:
	pip install -r requirements.txt

train:
	python scripts/train_model.py

predict:
	python scripts/generate_predictions.py

test:
	pytest tests
