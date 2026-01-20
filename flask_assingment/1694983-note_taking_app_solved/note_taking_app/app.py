from flask import Flask, render_template, request

# app = Flask(__name__)

# notes = []
# @app.route('/home', methods=["POST"]) # home

# def index():
#     note = request.args.get("note")
#     notes.append(note)
#     return render_template("home.html", notes=notes)


# if __name__ == '__main__':
#     app.run(debug=True)



app = Flask(__name__)

notes = []


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        note = request.form.get('note')
        notes.append(note)


    return render_template('home.html', notes=notes)




if __name__ == '__main__':
    app.run(debug=True)