.PHONY: style

check_dirs := oslo/ tests/

style:
	black $(check_dirs)
	isort $(check_dirs)
