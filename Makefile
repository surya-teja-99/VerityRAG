.PHONY: setup test demo eval clean

setup:
	pip3 install --break-system-packages torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
	pip3 install --break-system-packages -r requirements.txt
	python3 scripts/build_index.py

test:
	python3 -m pytest tests/ -q

demo:
	python3 scripts/demo.py

eval:
	python3 -m verityrag.eval --n_retrieval 500 --n_full 50

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache
