import string
import random
from urllib.parse import urlparse
from database import get_db_connection

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and parsed.netloc != ""
    except Exception:
        return False


def generate_short_code(length: int = 6) -> str:
    characters = string.ascii_letters + string.digits

    while True:
        short_code = "".join(random.choices(characters, k=length))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM urls WHERE short_code = ?",
            (short_code,)
        )
        exists = cursor.fetchone()
        conn.close()

        if not exists:
            return short_code
