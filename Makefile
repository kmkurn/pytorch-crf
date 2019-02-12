.PHONY: build upload test-upload

build:
	python setup.py bdist_wheel
	python setup.py sdist

upload: build
	twine upload --skip-existing dist/*

test-upload: build
	twine upload -r testpypi --skip-existing dist/*
