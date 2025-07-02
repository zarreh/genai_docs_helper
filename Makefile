lint: ## Lint and reformat the code
	@poetry run autoflake genai_docs_helper --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@poetry run black genai_docs_helper --line-length 120 -q
	@poetry run isort genai_docs_helper