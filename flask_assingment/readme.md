
# 📝 Regex Finder: Full-Stack Pattern Matcher

A powerful, user-friendly regex testing tool inspired by Regex101. This application leverages a **Flutter** frontend for a seamless user experience and a **Flask** backend for high-performance pattern matching.

## 🚀 Overview

This project clones the core functionality of professional regex testers. It allows users to input a test string and a regular expression, then dynamically fetches and displays all matches found using Python's robust `re` library.

### Key Features

* **Dynamic Matching:** Real-time extraction of matches from any test string.
* **Cross-Platform UI:** A clean, responsive interface built with Flutter.
* **Error Resilience:** Sophisticated error handling to catch and explain invalid regex syntax (e.g., unclosed groups or invalid escape characters).
* **RESTful Communication:** Seamless data exchange between the Flutter client and the Flask API.

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **Frontend** | [Flutter](https://flutter.dev/) (Dart) |
| **Backend** | [Flask](https://flask.palletsprojects.com/) (Python) |
| **Logic** | Python `re` Module |
| **API** | REST (HTTP POST/GET) |

---

## 📸 Screenshots

> `[c:\Users\admin\Pictures\Screenshots\Screenshot 2026-01-15 203949.png]`

---

## ⚙️ Installation & Setup

### 1. Backend (Flask)

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors

# Run the server
python app.py

```

### 2. Frontend (Flutter)

```bash
# Navigate to the frontend directory
cd frontend

# Get packages
flutter pub get

# Run the app
flutter run

```

---

## 💡 How it Works

1. The **Flutter** frontend captures the user's "Test String" and "Regex Pattern."
2. An API request is sent to the **Flask** server.
3. The backend processes the string using `re.finditer()` or `re.findall()`.
4. If the pattern is invalid, the backend returns a descriptive error message; otherwise, it returns a list of matches.
5. Flutter displays the results dynamically on the screen.

---

## 🎓 Acknowledgements

This project was developed as part of a task at **Innomatics Research Labs**. Special thanks to the mentors for the guidance on Flask backend logic and full-stack integration.
