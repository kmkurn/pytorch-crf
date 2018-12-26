build:
	python setup.py bdist_wheel

upload: build
	twine upload --skip-existing dist/*

.PHONY: build upload
