from distutils.log import debug 
from fileinput import filename 
from flask import * 
from flask import Flask, request, jsonify
from core import process

app = Flask(__name__)

@app.route('/')   
def main():   
    return render_template("index.html")   

@app.route("/upload", methods=["POST"])

def upload_document():
    # Get the uploaded file from the request
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # Save the uploaded file (replace with your desired storage solution)
    file.save(f"docs/{file.filename}")

    return "Document uploaded successfully!"

@app.route("/query", methods=["POST"])
def query_llm():
    question = request.json['question']
    response = process(question)
    print(response)
    return response

if __name__ == "__main__":
    app.run(debug=True) 