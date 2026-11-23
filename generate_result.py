
import json
from pathlib import Path

import gymnasium
import numpy as np
import pandas as pd

from q_learning import (
    deploy,
    SupplyData,
    DigitalHealthEnv,
    DQNStrategyWrapper
)


# --------------------------------------------------
# 1. LOAD YOUR EXISTING DATASETS
# --------------------------------------------------

df1 = pd.read_csv(
    "C:/Users/VP/Downloads/archive SCD/drugs.csv"
)

df2 = pd.read_csv(
    "C:/Users/VP/Downloads/archive SCD/stores.csv"
)

df3 = pd.read_csv(
    "C:/Users/VP/Downloads/archive SCD/dispensing.csv"
)

n_weeks = int(df3["week"].max()) + 1

supply_data = SupplyData(
    path=(df1, df2, n_weeks)
)

df = pd.read_csv(
    "C:/Users/VP/Downloads/archive EHR/EHR.csv"
)

df4 = pd.read_csv(
    "C:/Users/VP/Downloads/archive SCD/context.csv"
)

visitor = pd.read_csv(
    "C:/Users/VP/Downloads/archive (4)/visits_encounters.csv",
    parse_dates=["admission_datetime"]
)

clini_detail = pd.read_csv(
    "C:/Users/VP/Downloads/archive (4)/clinical_details.csv"
)

df5 = visitor.merge(
    clini_detail[
        [
            "encounter_id",
            "disease_category",
            "clinical_record_id",
            "disease_severity",
            "primary_diagnosis_name"
        ]
    ],
    on="encounter_id"
)

df6 = pd.read_csv(
    "C:/Users/VP/Downloads/archive IoTBWD/AIoT_Orienteering_Dataset/AIoT_Orienteering_Full_Dataset.csv"
)

df7 = df6.sort_values(
    ["participant_id", "session_id", "timestamp"]
)

df5["admission_datetime"] = pd.to_datetime(
    df5["admission_datetime"]
)

df5["week"] = (
    df5["admission_datetime"]
    .dt.to_period("W")
    .apply(lambda x: x.start_time)
)


# --------------------------------------------------
# 2. PREPARE ENVIRONMENT DATA
# --------------------------------------------------

data = {
    "ehr": df,
    "clinical": df5,
    "context": df4,
    "iot": df7
}


# --------------------------------------------------
# 3. REGISTER AND INITIALIZE ENVIRONMENT
# --------------------------------------------------

gymnasium.register(
    id="DigitalHealth-v0",
    entry_point=DigitalHealthEnv,
    kwargs={
        "data": data,
        "supply_data": supply_data
    }
)

env = gymnasium.make("DigitalHealth-v0")

env.reset()

base_env = env.unwrapped


# --------------------------------------------------
# 4. GET OBSERVATION
# --------------------------------------------------

observation = base_env.get_observation()

for key, value in observation.items():
    print(key, np.asarray(value).shape)


# --------------------------------------------------
# 5. PPO PREDICTION
# --------------------------------------------------

investment_actions = deploy(
    observation,
    base_env.observation_space
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


# --------------------------------------------------
# 6. Q-LEARNING OPERATIONAL DECISIONS
# --------------------------------------------------

action_converter = DQNStrategyWrapper(env)

_, reward, terminated, truncated, info = (
    action_converter.step(investment_actions)
)

base_env = env.unwrapped


# --------------------------------------------------
# 7. COLLECT DASHBOARD RESULTS
# --------------------------------------------------

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
    ),

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

    "operational_results": info.get(
        "pair_results", []
    )
}


# --------------------------------------------------
# 8. CONVERT NUMPY/PANDAS VALUES FOR JSON
# --------------------------------------------------

def make_json_serializable(value):

    if isinstance(value, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    return value


dashboard_data = make_json_serializable(
    dashboard_data
)


# --------------------------------------------------
# 9. SAVE RESULTS FOR VERCEL
# --------------------------------------------------

results_file = (
    Path(__file__).resolve().parent
    / "investment_results.json"
)

with open(results_file, "w", encoding="utf-8") as file:
    json.dump(
        dashboard_data,
        file,
        indent=4,
        allow_nan=False
    )

print("\nPPO recommendation results generated successfully.")

print("Investment actions:", investment_actions)

print("Total reward:", reward)

print("Results saved to:", results_file)

env.close()
