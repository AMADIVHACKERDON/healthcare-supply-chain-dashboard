import json
from pathlib import Path

from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder="templates")

ACTIONS = {
     0: "hold",
    1: "small_order",
    2: "medium_order",
    3: "large_order",
    4: "expedite_small",
    5: "expedite_medium",
    6: "expedite_large",
    7: "capital_preservation"
}
RESULTS_FILE = (
    Path(__file__).resolve().parent
    / "investment_results.json"
)



@app.route("/")
def index():
    return render_template("index.html")


@app.get("/recommend")
def recommend():
    if not RESULTS_FILE.exists():
        return jsonify({
            "error": "No PPO recommendation results available."
        }), 503

    with open(RESULTS_FILE, "r") as file:
        results = json.load(file)

    return jsonify(results)
    

    
@app.route("/investment-history")
def investment_history():
    import json
    import os

    history_path = os.path.join(
        app.root_path,
        "investment_history.json"
    )

    if not os.path.exists(history_path):
        return jsonify([])

    with open(history_path, "r") as file:
        history = json.load(file)

    return jsonify(history)

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False)
