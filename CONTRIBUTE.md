# Contribution Notes

## Setup
```
python -m venv test-env --upgrade-deps
source test-env/bin/activate
pip install -e ".[dev,test]"
rehash
pre-commit install
```
