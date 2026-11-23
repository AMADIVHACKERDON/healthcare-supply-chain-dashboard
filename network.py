
import flask
from flask import Flask, render_template, request

from q_learning import deploy
from deployment.strategy import strategy_to_action

app = Flask(__name__, template_folder='templates')

ACTIONS = {
    0: "hold",
    1: "small_order",
    2: "medium_order",
    3: "large_order",
    4: "risk_based_order",
    5: "disease_based_order",
    6: "emergency_order",
    7: "capital_preservation"
}



@app.route('/')
def index():
    return render_template('index.html')

@app.post("/recommend")
def recommend():

    data = request.get_json()

    observation = data["observation"]

    strategy = deploy(observation)

    return {
        "strategy": strategy,
        "strategy_name": ACTIONS[strategy]
    }
if __name__ == "__main__":
    app.run(debug=True)
