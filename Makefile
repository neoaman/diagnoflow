build:
	virtualenv .venv --prompt . -p python3.11
	.venv/bin/python -m pip install -r requirements.txt

install:
	.venv/bin/python -m pip install -r requirements.txt

run:
	.venv/bin/python -m gunicorn app:server -b 0.0.0.0:8050 --reload
# https://dev-dash.mlhub.in