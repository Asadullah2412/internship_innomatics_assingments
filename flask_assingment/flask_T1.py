from flask import Flask,request

app = Flask(__name__)

@app.route('/upname') # uppername
def hello_user():
    name = request.args.get('name')
    if not name:
        return "Please provide a name using ?name=your name"
    
    upper_name = name.upper()
    return f'Hello {upper_name}'

@app.route('/lowname')
def bye_user():
    name = request.args.get('name')
    if not name:
        return "Please provide a name using ?name = your name"
    
    lower_name = name.lower()
    return f'Bye {lower_name}'
if __name__ == '__main__':
    app.run(debug=True)