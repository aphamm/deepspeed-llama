import time

from flask import Flask, jsonify, render_template, request

from inference import generate_response, load_model

app = Flask(__name__)

model_path = "aphamm/stage1"
model, tokenizer = load_model(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.form["user_input"]
    start = time.time()
    response = generate_response(model, tokenizer, user_input)
    return jsonify({"response": response, "time": time.time() - start})


if __name__ == "__main__":
    app.run(port=5001, debug=False)
