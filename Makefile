build:
	python setup.py bdist_wheel

upload: build
	twine upload dist/*

.PHONY: build upload
