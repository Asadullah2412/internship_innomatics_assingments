from flask import Flask, jsonify
from flask_cors import CORS
from config import DEBUG
from database import init_db

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def health_check():
        return jsonify({"status": "Flask backend running"})

    return app


if __name__ == "__main__":
    init_db()
    app = create_app()
    app.run(debug=DEBUG)
