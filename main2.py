import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
from q_learning import deploy
from q_learning import SupplyData
from deployment.strategy import strategy_to_action
from q_learning import (
    DigitalHealthEnv,
    ACTIONS
)
from q_learning import DQNStrategyWrapper

app = Flask(__name__, template_folder="templates"
            )

df1 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/drugs.csv")
df2 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/stores.csv")
df3 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/dispensing.csv")
n_weeks = int(df3["week"].max()) + 1

supply_data = SupplyData(
    path=(df1 ,df2, n_weeks)
)

df = pd.read_csv("C:/Users/VP/Downloads/archive EHR/EHR.csv")
df4 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/context.csv")
visitor = pd.read_csv("C:/Users/VP/Downloads/archive (4)/visits_encounters.csv",parse_dates=["admission_datetime"])
clini_detail = pd.read_csv("C:/Users/VP/Downloads/archive (4)/clinical_details.csv")
df5 = visitor.merge(clini_detail[['encounter_id','disease_category',  "clinical_record_id","disease_severity","primary_diagnosis_name"]], on='encounter_id')
df6 = pd.read_csv("C:/Users/VP/Downloads/archive IoTBWD/AIoT_Orienteering_Dataset/AIoT_Orienteering_Full_Dataset.csv")
df7 = df6.sort_values(["participant_id", "session_id", "timestamp"])
df5["admission_datetime"] = pd.to_datetime(df5["admission_datetime"])
df5["week"] = (df5["admission_datetime"].dt.to_period("W").apply(lambda x: x.start_time))
demand_data = (
    df5.groupby(["week", "disease_category"])
       .size()
       .reset_index(name="admission_count")
)
print("Hospital demand data:")
print(demand_data.head())
weekly_disease_demand = (df5.groupby(["week", "disease_category"]).size().reset_index(name="disease_cases"))

data = {
    "ehr": df,
    "clinical": df5,
    "context": df4,
    "iot": df7
}
env = DigitalHealthEnv(
    data=data,
    supply_data=supply_data,
    demand_data=demand_data
)


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


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/recommend")
def recommend():
    base_env = env.unwrapped

    observation = base_env.get_observation()

    for key, value in observation.items():
        print(key, np.asarray(value).shape)

    # PPO predicts the four investment allocations.
    investment_actions = deploy(
        observation,
        base_env.unwrapped.observation_space
    )

    investment_actions = np.asarray(
        investment_actions,
        dtype=np.float32
    ).reshape(-1)

    if investment_actions.shape != (4,):
        raise ValueError(
            f"Expected four investment actions, "
            f"received {investment_actions.shape}"
        )

    # Q-learning generates the operational decisions.
    action_converter = DQNStrategyWrapper(env)


    _, reward, terminated, truncated, info = (
        action_converter.step(investment_actions)
    )
    base_env = env.unwrapped

    dashboard_data = {
        "initial_capital": float(
            base_env.initial_balance
        ),

        "cumulative_invested": float(
            base_env.invested_capital
        ),

        "capital_remaining": float(
            base_env.investment_capital
        ),

        "latest_investment_reward": float(
            info.get("investment_reward", 0.0)
        )
    }

    return jsonify({
        **dashboard_data,

        "investment_actions": [
            float(x) for x in investment_actions
        ],

        "investment_by_area": {
            str(k): float(v)
            for k, v in base_env.investment_by_area.items()
        },

        "investment_reward": float(
            info.get("investment_reward", 0.0)
        ),

        "reward": float(reward),

        "terminated": bool(terminated),
        "truncated": bool(truncated),

        "operational_results": [
            {
                str(k): (
                    float(v)
                    if isinstance(v, (np.integer, np.floating))
                    else v
                )
                for k, v in result.items()
            }
            for result in info.get("pair_results", [])
        ]
    })


    
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
