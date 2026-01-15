from flask import Flask,request,jsonify
import re

app = Flask(__name__)

@app.route('/match',methods=['POST'])
def match_regex():
    data = request.get_json()

    test_string = data.get('test_string')
    pattern = data.get('pattern')

    if not  test_string or not pattern:
        return jsonify({"error": "test_string and pattern are required"}),400
    try:
        matches = re.findall(pattern,test_string)
        return jsonify({'matches':matches})
    except re.error as e:
        return jsonify({"error":f"invalid regex: {str(e)}"}),400
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)