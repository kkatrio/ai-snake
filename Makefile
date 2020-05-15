.PHONY: train test run

train: 
	./train_snake.py

test: export PYTHONPATH=.
test:
	pytest tests/test_snake.py --disable-pytest-warnings

run:
	./run_evolved_snake.py trained_snake.model
