install: ## Install required dependencies
	poetry install --no-dev

install-dev: ## Install required dependencies plus dev packages
	poetry install

unit: ## Run all unit tests
	poetry run pytest src/test/unit/

coverage: ## Run all unit tests with coverage report
	poetry run pytest src/test/unit --cov=src/main

functional: ## Run all functional tests
	cd src && ./run_functional_tests.sh