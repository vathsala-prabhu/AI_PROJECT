"""Settings loaded from the .env file in the project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
# override=True: the .env file wins over any DATABASE_URL set in Windows
load_dotenv(ENV_PATH, override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")
TOKEN_EXPIRE_MINUTES = 60