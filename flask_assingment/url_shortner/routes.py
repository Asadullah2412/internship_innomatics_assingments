from flask import Blueprint, request, jsonify, redirect
from database import get_db_connection
from utils import is_valid_url, generate_short_code

routes = Blueprint("routes", __name__)

@routes.route("/api/shorten", methods=["POST"])
def shorten_url():
    data = request.get_json()

    if not data or "long_url" not in data:
        return jsonify({"error": "URL is required"}), 400

    long_url = data["long_url"].strip()

    if not is_valid_url(long_url):
        return jsonify({"error": "Invalid URL"}), 400

    short_code = generate_short_code()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO urls (original_url, short_code) VALUES (?, ?)",
        (long_url, short_code)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "short_url": f"http://127.0.0.1:5000/{short_code}"
    }), 201


@routes.route("/<short_code>", methods=["GET"])
def redirect_to_url(short_code):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT original_url, clicks FROM urls WHERE short_code = ?",
        (short_code,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({"error": "Short URL not found"}), 404

    cursor.execute(
        "UPDATE urls SET clicks = clicks + 1 WHERE short_code = ?",
        (short_code,)
    )
    conn.commit()
    conn.close()

    return redirect(row["original_url"])
