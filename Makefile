lint: ## Lint and reformat the code
	@poetry run autoflake genai_docs_helper --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@poetry run black genai_docs_helper --line-length 120 -q
	@poetry run isort genai_docs_helper

ingest: ## Install dependencies
	@poetry run python -m genai_docs_helper.loader_embed_to_vectorstore

run: ## Run the application
	@poetry run python -m genai_docs_helper.graph

graph: ## Generate the graph
	@poetry run langgraph dev

docs: ## Generate HTML documentation
	@mkdir -p docs
	@poetry run pdoc genai_docs_helper -o docs -d google --show-source
	@echo "Documentation generated in docs/"

docs-serve: ## Generate and serve documentation
	@poetry run pdoc genai_docs_helper -d google --show-source

docs-clean: ## Clean generated documentation
	@rm -rf docs/