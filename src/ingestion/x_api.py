import requests
import os
import logging
import time
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def fetch_x_posts(query="BTC"):
    api_token = os.getenv("X_API_TOKEN")
    if not api_token:
        logger.error("X_API_TOKEN no configurada en .env")
        return None
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(
            f"https://api.x.com/2/tweets/search/recent?query={query}", headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"X posts: {data}")
        time.sleep(1)  # Retardo para evitar 429
        return data
    except Exception as e:
        logger.error(f"Error en X API: {e}")
        return None


if __name__ == "__main__":
    for _ in range(2):  # Prueba 2 llamadas
        fetch_x_posts()
