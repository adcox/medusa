# Contribution Notes

## Setup
```
python -m venv test-env --upgrade-deps
source test-env/bin/activate
pip install -e ".[dev,test]"
rehash
pre-commit install
```

## Testing
```
source test-env/bin/activate
pytest --cov medusa --cov-report html
mypy
```

## Documentation
```
source test-env/bin/activate
cd docs
make html
```