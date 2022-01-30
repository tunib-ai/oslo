.PHONY: style

check_dirs := oslo/ tests/ tutorials/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)