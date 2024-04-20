# Python, FastAPI, Docker

This project houses the model API for Signbuddy.

To run this project do the following:

1. Create a python virtual environment; `python -m venv venv`
2. Activate venv; `.\venv\Scripts\activate` on windows
3. Run `pip install -r requirements.txt`
4. Create .env file with necessary environment variables.
5. Run `uvicorn app.main:app --reload`

## References

- All code was custom developed or was developed by following tutorials from official sources such as FastAPI, Python and Docker