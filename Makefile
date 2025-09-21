lint: ## Lint and reformat the code
	@echo "Removing unused imports and variables..."
	@poetry run autoflake genai_docs_helper --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@echo "Formatting code with Black..."
	@poetry run black genai_docs_helper --line-length 120 -q
	@echo "Sorting imports with isort..."
	@poetry run isort genai_docs_helper

ingest: ## Install dependencies
	@echo "Loading and embedding documents to vectorstore..."
	@poetry run python -m genai_docs_helper.loader_embed_to_vectorstore

run: ## Run the application
	@echo "Starting the GenAI docs helper application..."
	@poetry run python -m genai_docs_helper.graph

graph: ## Generate the graph
	@echo "Starting LangGraph development server..."
	@poetry run langgraph dev

docs: ## Generate HTML documentation
	@echo "Creating docs directory..."
	@mkdir -p docs
	@echo "Generating HTML documentation with pdoc..."
	@poetry run pdoc genai_docs_helper -o docs -d google --show-source
	@echo "Documentation generated in docs/"

docs-serve: ## Generate and serve documentation
	@echo "Starting documentation server..."
	@poetry run pdoc genai_docs_helper -d google --show-source

docs-clean: ## Clean generated documentation
	@echo "Cleaning generated documentation..."
	@rm -rf docs/

test: ## Run all tests
	@echo "Running all tests..."
	@poetry run pytest

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	@poetry run pytest tests/unit -v

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	@poetry run pytest tests/integration -v -m integration

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage report..."
	@poetry run pytest --cov=genai_docs_helper --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	@echo "Starting test watcher..."
	@poetry run pytest-watch

test-parallel: ## Run tests in parallel
	@echo "Running tests in parallel..."
	@poetry run pytest -n auto

test-verbose: ## Run tests with verbose output
	@echo "Running tests with verbose output..."
	@poetry run pytest -vvs

test-specific: ## Run specific test file (usage: make test-specific TEST=tests/unit/nodes/test_retrieve.py)
	@echo "Running specific test: $(TEST)"
	@poetry run pytest $(TEST) -v