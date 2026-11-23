import gymnasium
from gymnasium import spaces
from gymnasium.spaces import flatten
from gymnasium.spaces.utils import flatten
from gymnasium.envs.registration import register
import numpy as np
import random
from enum import Enum
import pandas as pd

from deployment.strategy import strategy_to_action

import stable_baselines3
from stable_baselines3 import PPO
import time
import torch

SIGNALS = ["latitude", "longitude", "altitude", "speed","acc_x", "acc_y", "acc_z","gyro_x", "gyro_y", "gyro_z", "heart_rate"]

EXPEDITE_COST = 0.20       
EXPEDITE_REDUCTION = 2   
MIN_EXPEDITE_LEAD_TIME = 1
COST_RATIO = 0.70
START = pd.Timestamp("2023-01-02")

CATEGORY_TO_DRUGS = {
    "Respiratory": [
        "D10", "D11", "D12", "D14", "D19"],

    "Infectious": ["D08", "D11", "D12", "D16"],

    "Cardiovascular": ["D03", "D04", "D05"],

    "Endocrine": ["D01", "D02", "D07"],

    "Gastrointestinal": ["D09", "D17", "D18"]
}

class SupplyData:
    
     def __init__(self, path, hr_high=100.0, assign_wearables = True, seed=0):
        super().__init__()
        df1 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/drugs.csv")
        df2 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/stores.csv")
        df3 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/dispensing.csv")
        df4 = pd.read_csv("C:/Users/VP/Downloads/archive SCD/context.csv")
        visitor = pd.read_csv("C:/Users/VP/Downloads/archive (4)/visits_encounters.csv",parse_dates=["admission_datetime"])
        clini_detail = pd.read_csv("C:/Users/VP/Downloads/archive (4)/clinical_details.csv")
        df5 = visitor.merge(clini_detail[['encounter_id','disease_category',  "clinical_record_id","disease_severity","primary_diagnosis_name"]], on='encounter_id')
        df6 = pd.read_csv("C:/Users/VP/Downloads/archive IoTBWD/AIoT_Orienteering_Dataset/AIoT_Orienteering_Full_Dataset.csv")
        df7 = df6.sort_values(["participant_id", "session_id", "timestamp"])
        self.drugs, self.stores = df1, df2
        self.stockout_features = self.build_stockout_features(df3)
        print(self.stockout_features[["store_id","drug_id","week","stock_on_hand","forecast_demand","inventory_coverage","stockout_risk"]].head(10))
        
        # df2 = pd.read_csv("C:/Users/VP/Downloads/archive SuD")
        # df3 = pd.read_csv("C:/Users/VP/Downloads/archive BWD")
        
        self.n_drugs, self.n_stores = len(df1), len(df2)
        self.n_weeks = int(df3["week"].max()) + 1
        
        self.drug_ids = df1["drug_id"].tolist()
        self.store_id = df2['store_id'].tolist()
        self.build_disease_demand(df5)


        self.si = {s: i for i, s in enumerate(self.store_id)}
        self.di = {d: i for i, d in enumerate(self.drug_ids)}
        s_i = df3["store_id"].map(self.si).to_numpy()
        d_i = df3["drug_id"].map(self.di).to_numpy()
        w_i = df3["week"].to_numpy()

     
        def cube(col):
            a = np.zeros((self.n_stores, self.n_drugs, self.n_weeks), dtype=np.float32)
            a[s_i, d_i, w_i] = df3[col].to_numpy()
            return a

        self.demand = cube("demand")
        self.price = cube("unit_price_inr")
        self.stock = cube("stock_on_hand")
        self.cost = self.price * COST_RATIO
        self.baseline = df1["baseline_weekly_demand"].to_numpy(dtype=np.float32)
        self.lead_time = df2["supplier_lead_time_weeks"].to_numpy(dtype=np.int32)
        self.participant_ids = sorted(df7["participant_id"].unique().tolist())
        self.n_participants = len(self.participant_ids)
        self.n_sessions = int(df7["session_id"].max())
        self.n_seconds = int(df7["timestamp"].max())+1
        
        rng = np.random.default_rng(seed)
        self.cohort_assignment = rng.integers(0, self.n_stores, size=self.n_participants)
       
        self.cohort_load = np.zeros(self.n_stores, dtype=np.float32)
        self.cohort_hr_high = np.zeros(self.n_stores, dtype=np.float32)
        self.cohort_size = np.zeros(self.n_stores, dtype=np.int32)


        
        pi = {p: i for i, p in enumerate(self.participant_ids)}
        p_idx = df6["participant_id"].map(pi).to_numpy()
        s_idX = df6["session_id"].to_numpy() - 1
        t_idx = df6["timestamp"].to_numpy()

        shape = (self.n_participants, self.n_sessions, self.n_seconds)
        self.signals = {}
        for name in SIGNALS:
            a = np.zeros(shape, dtype=np.float32)
            a[p_idx,s_idX,t_idx] = df6[name].to_numpy()
            self.signals[name] = a

        terrain = np.zeros(shape, dtype=np.float32)
        terrain[p_idx, s_idX, t_idx] = df6["terrain_type"].to_numpy()
        self.terrain = terrain
        self.hr_high = hr_high
        
        self.assign_wearables = assign_wearables
        self.build_session()
        self.assign_cohorts()
        self.build_wearable_signals()


     def build_stockout_features(self, df3):
          df = df3.sort_values(["store_id", "drug_id", "week"]).copy()
          # 1. Identify demand that could not be fulfilled
          df["unmet_demand"] = (df["demand"] - df["units_dispensed"]).clip(lower=0)
          # 2. Calculate historical stockout frequency
          # Uses the current and previous weeks, up to 4 weeks
          df["stockout_history"] = (
          df.groupby(["store_id", "drug_id"])["stockout"].transform(lambda x:x.shift(1).rolling(
                window=4, min_periods=1
            ).mean()
        )
    ).fillna(0)

    # 3. Calculate historical unmet-demand rate
          df["unmet_demand_history"] = (df.groupby(["store_id", "drug_id"])["unmet_demand"].transform(
            lambda x: x.shift(1).rolling(
                window=4, min_periods=1
            ).mean()
        )
    ).fillna(0)

    # 4. Estimate next week's demand using the
    # average demand from the latest 4 recorded weeks
          df["forecast_demand"] = (df.groupby(["store_id", "drug_id"])["demand"].transform(
            lambda x: x.shift(1).rolling(
                window=4, min_periods=1
            ).mean()
        )
    )
          previous_demand = (
               df.groupby(["store_id", "drug_id"])["demand"].transform(
                    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()))

          earlier_demand = (
               df.groupby(["store_id", "drug_id"])["demand"].transform(
                    lambda x: x.shift(5).rolling(window=4, min_periods=1).mean()))

          df["demand_trend"] = np.select([
               previous_demand > earlier_demand * 1.1,
               previous_demand < earlier_demand * 0.9
               ],
               [2, 0],
               default=1 )

    # 5. Attach supplier lead time to each store
          lead_time = self.stores[["store_id", "supplier_lead_time_weeks"]]
          df = df.merge(
        lead_time,
        on="store_id",
        how="left"
    )

    # 6. Estimate demand during supplier lead time
          df["lead_time_demand"] = (df["forecast_demand"] *df["supplier_lead_time_weeks"])

    # 7. Calculate inventory coverage
          df["inventory_coverage"] = (df["stock_on_hand"] /df["lead_time_demand"].clip(lower=1))

    # 8. Classify stockout risk
          df["stockout_risk"] = np.select(
        [
            df["inventory_coverage"] < 0.5,
            df["inventory_coverage"] < 1.0,
            df["inventory_coverage"] < 1.5
        ],
        [
            3,  # Critical risk
            2,  # High risk
            1   # Moderate risk
        ],
        default=0  # Low risk
    )

          self.stockout_features = df

          return self.stockout_features
          

     def build_session(self):
         hr = self.signals["heart_rate"]
         acc = np.sqrt(self.signals["acc_x"] ** 2 + self.signals["acc_y"] ** 2 + self.signals["acc_z"] ** 2)
         gyro = np.sqrt(self.signals["gyro_x"] ** 2 + self.signals["gyro_y"] ** 2 + self.signals["gyro_z"] ** 2)

         self.mean_hr = hr.mean(axis=2)
         self.max_hr = hr.max(axis=2)
         self.frac_hr_high = (hr > self.hr_high).mean(axis=2)
         self.mean_speed = self.signals["speed"].mean(axis=2)
         self.motion_intensity = acc.mean(axis=2)
         self.gyro_intensity = gyro.mean(axis=2)
         self.centroid_lat = self.signals["latitude"].mean(axis=2)
         self.centroid_lon = self.signals["longitude"].mean(axis=2)
         self.session_terrain = np.round(self.terrain.mean(axis=2)).astype(np.int8)
         self.load_index = np.clip((self.mean_hr - 60.0) / 60.0, 0.0, 1.0)
         self.per_part_load = self.load_index.mean(axis=1)
         self.per_part_high = self.frac_hr_high.mean(axis=1)
     
     def assign_cohorts(self):
          if not self.assign_wearables:
               return
          for s in range(self.n_stores):
               members = np.where(self.cohort_assignment == s)[0]
               self.cohort_size[s] = len(members)
               if len(members):
                    self.cohort_load[s] = self.per_part_load[members].mean()
                    self.cohort_hr_high[s] = self.per_part_high[members].mean()

     def build_wearable_signals(self):
          sig = np.full((self.n_stores, self.n_weeks), np.nan, dtype=np.float32)
          session_of_week = np.arange(self.n_weeks) % self.n_sessions
          for st in range(self.n_stores):
               members = np.where(self.cohort_assignment == st)[0]
               if len(members):
                    per_session = self.frac_hr_high[members].mean(axis=0)
                    sig[st] = per_session[session_of_week]
          sig = np.where(np.isnan(sig), np.nanmedian(sig), sig).astype(np.float32)
          self.wearable_signal = np.clip(sig, 0.0, 1.0)
          cuts = np.quantile(self.wearable_signal, np.linspace(0,1, 3+1)[1:-1])
          self.wearable_cuts = cuts
          self.wearable_state = np.digitize(self.wearable_signal, cuts).astype(np.int8)
    
          

     def session(self, participant_id, session_id):
         p = self.participant_ids.index(participant_id)
         s = session_id - 1
         result = dict(mean_hr = float(self.mean_hr[p,s].mean()),
                     max_hr = float(self.max_hr[p,s].max()),
                     frac_hr_high = float(self.frac_hr_high[p,s].mean()),
                     mean_speed = float(self.mean_speed[p,s].mean()),
                     motion_intensity = (self.motion_intensity[p,s].mean()),
                     load_index = float(self.load_index[p,s].mean()))
        
         return result
         
     def summary(self):
         print(f"participants={self.n_participants} sessions={self.n_sessions} "
               f"seconds={self.n_seconds}")
         print("mean HR:", round(float(self.mean_hr.mean()), 1),
                "Heart rate threshold:", self.hr_high,
               "share of time:",f"{self.frac_hr_high.mean():.1%}")
         print("load index:", round(float(self.load_index.mean()), 2),
               round(float(self.load_index.min()), 2), round(float(self.load_index.max()),2))
     def build_disease_demand(self, df5):
                  df5["week"] = ((df5["admission_datetime"].dt.normalize() - START).dt.days // 7)

                  # Keep only encounters within the dispensing timeline
                  clinical = df5[(df5["week"] >= 0) & (df5["week"] < self.n_weeks)].copy()
     
         # Count weekly disease cases
                  weekly_cases = (clinical.groupby(["disease_category", "week"]).size()
                                  .unstack(fill_value=0).reindex(columns=range(self.n_weeks),
                                                                 fill_value=0))
     
                  self.disease_categories = weekly_cases.index.tolist()
     
         # Store disease cases
                  self.weekly_disease_cases = weekly_cases.to_numpy(dtype=np.float32)
     
         # Distribute estimated cases across stores
                  populations = self.stores["catchment_population"].to_numpy(dtype=np.float32)
                  population_weights = (populations /max(populations.sum(), 1))
     
                  self.store_disease_cases = (population_weights[:, None, None] * self.weekly_disease_cases[None, :, :])
     
         # Create drug-to-disease-category relationship
                  self.drug_category_link = np.zeros(
                       (self.n_drugs, len(self.disease_categories)),dtype=np.float32)
     
                  for drug_idx, drug_id in enumerate(self.drug_ids):
                       for category_idx, category in enumerate(
                            self.disease_categories):
                            associated_drugs = CATEGORY_TO_DRUGS.get(category, []
                                                                     )
                            if drug_id in associated_drugs:
                                self.drug_category_link[
                         drug_idx, category_idx] = 1.0
                  print("\nDISEASE-DRUG LINK DIAGNOSTICS")
                  print("Number of links:", int(self.drug_category_link.sum()))

                  for category_idx, category in enumerate(self.disease_categories):
                        linked_drugs = [
                    self.drug_ids[drug_idx]
for drug_idx in range(self.n_drugs)
if self.drug_category_link[drug_idx, category_idx] == 1
]

                  print(category, ":", linked_drugs)

         # Calculate disease pressure
                  
                  self.disease_pressure = np.ones(
                        (self.n_stores, self.n_drugs, self.n_weeks),
                        dtype=np.float32
                    )

                    # Process each disease category
                  for category_idx in range(len(self.disease_categories)):

                        cases = weekly_cases.iloc[category_idx]

                        # Use previous weeks only
                        previous_cases = cases.shift(1)

                        previous_average = (
                            cases.shift(1)
                            .rolling(window=4, min_periods=2)
                            .mean()
                        )

                        pressure = (
                            previous_cases /
                            previous_average.replace(0, np.nan)
                        )

                        pressure = pressure.fillna(1.0)

                        pressure = pressure.clip(
                            lower=0.5,
                            upper=2.0
                        )

                        print(
                            self.disease_categories[category_idx],
                            "Weekly cases:", cases.min(), cases.max(),
                            "Pressure:", pressure.min(), pressure.max(),
                            "Mean:", pressure.mean()
                        )

                        # Apply this disease pressure to its associated drugs
                        for drug_idx in range(self.n_drugs):

                            if self.drug_category_link[drug_idx, category_idx] == 1:

                                self.disease_pressure[:, drug_idx, :] *= (
                                    pressure.to_numpy(dtype=np.float32)
                                )

                    # Clip combined effects after all categories have been processed
                  self.disease_pressure = np.clip(
                        self.disease_pressure,
                        0.5,
                        2.0
                    )

                    # Final diagnostics
                  print("\nDisease-demand feature created.")
                  print("Disease categories:", self.disease_categories)
                  print("Disease-pressure shape:", self.disease_pressure.shape)

                  print("\n========== DISEASE DEMAND DIAGNOSTICS ==========")

                  print("Number of disease categories:", len(self.disease_categories))

                  print(
                        "Number of drug-category links:",
                        np.count_nonzero(self.drug_category_link)
                    )

                  print("Disease link matrix shape:", self.drug_category_link.shape)

                  print("Weekly disease cases:")
                  print("Minimum:", self.weekly_disease_cases.min())
                  print("Maximum:", self.weekly_disease_cases.max())

                  print("Disease pressure:")
                  print("Minimum:", self.disease_pressure.min())
                  print("Maximum:", self.disease_pressure.max())
                  print("Mean:", self.disease_pressure.mean())
                  print("\nPressure by linked drug:")

                  for drug_idx, drug_id in enumerate(self.drug_ids):

                        if self.drug_category_link[drug_idx].sum() > 0:

                            values = self.disease_pressure[:, drug_idx, :]

                            print(
                                drug_id,
                                "Minimum:", values.min(),
                                "Maximum:", values.max(),
                                "Mean:", values.mean()
                            )
class DigitalHealthEnv(gymnasium.Env):
     metadata = {"render_modes": ["human"]}
     def __init__(self, data, supply_data,demand_data=None, initial_balance=1_000_000, render_mode = None,use_wearable=True):
          super(DigitalHealthEnv,self).__init__()
     
         
          self.supply_data = supply_data
          self.demand_data = demand_data
          self.use_wearable = use_wearable

          self.data = (
            supply_data.drugs,
            supply_data.stores
        )
          self.n_drugs = self.supply_data.n_drugs
          self.n_stores = self.supply_data.n_stores
          self.inventory = self.supply_data.stock[:, :,0].copy()
          
          self.action_space = spaces.Box(low=0.0,high=1.0,shape=(4 + self.n_drugs + (self.n_drugs * self.n_stores),),dtype=np.float32)
          self.initial_balance = initial_balance
          # Investment accounting
          self.invested_capital = 0.0
          self.investment_areas = [
    "forecasting",
    "inventory_management",
    "distribution",
    "supply_chain_capacity"
]

          self.investment_by_area = {
              "forecasting": 0.0,
              "inventory_management": 0.0,
              "distribution": 0.0,
              "supply_chain_capacity": 0.0
          }
          self.technology_capabilities = {
      area: 0.0 for area in self.investment_areas
  }
          self.previous_stockout_rate = 0.0
          self.previous_unmet_demand = 0.0
          self.previous_service_level = 0.0
          self.observation_space = spaces.Dict({"stock_on_hand": spaces.Box(low=0,high=np.inf,shape=(self.n_stores, self.n_drugs),dtype=np.float32), "forecast_demand": spaces.Box(low=0,high=np.inf, shape=(self.n_stores, self.n_drugs), dtype=np.float32), "stockout_history": spaces.Box(low=0, high=1, shape=(self.n_stores, self.n_drugs),dtype=np.float32), "unmet_demand_history": spaces.Box(low=0,high=np.inf,shape=(self.n_stores, self.n_drugs),dtype=np.float32),"inventory_coverage": spaces.Box(low=0,high=np.inf,shape=(self.n_stores, self.n_drugs),dtype=np.float32),"stockout_risk": spaces.Box(low=0,high=3,shape=(self.n_stores, self.n_drugs),dtype=np.int32),"investment_capital": spaces.Box(low=0,high=self.initial_balance,shape=(1,),dtype=np.float32),"disease_demand": spaces.Box(low=0,high=np.inf,shape=(self.n_stores, self.n_drugs),dtype=np.float32),"wearable_load":spaces.Box(low=0,high=1,shape=(self.n_stores,),dtype=np.float32),})
          
          self.reset()
          self.stockout_lookup = (self.supply_data.stockout_features.set_index(["store_id", "drug_id", "week"]).sort_index())
          # Precompute forecast demand for fast NumPy access.
          sf = self.supply_data.stockout_features

          store_to_idx = {
                store_id: i
                for i, store_id in enumerate(self.supply_data.store_id)
            }

          drug_to_idx = {
                drug_id: i
                for i, drug_id in enumerate(self.supply_data.drug_ids)
            }

            # Use baseline demand as the fallback.
          self.forecast_cube = np.broadcast_to(
                self.supply_data.baseline[None, :, None],
                (
                    self.n_stores,
                    self.n_drugs,
                    self.supply_data.n_weeks
                )
            ).copy()

          store_indices = sf["store_id"].map(store_to_idx).to_numpy(dtype=np.int32)
          drug_indices = sf["drug_id"].map(drug_to_idx).to_numpy(dtype=np.int32)
          week_indices = sf["week"].to_numpy(dtype=np.int32)

          forecast_values = pd.to_numeric(
               sf["forecast_demand"],
               errors="coerce"
           ).to_numpy(dtype=np.float32)

          valid = (
              np.isfinite(forecast_values)
              & (forecast_values > 0)
          )

          self.forecast_cube[
              store_indices[valid],
              drug_indices[valid],
              week_indices[valid]
          ] = forecast_values[valid]

          print("Forecast lookup array created:", self.forecast_cube.shape)
     def get_observation(self):
          # Get records for the current week
          df = self.supply_data.stockout_features
          current_data = df[df["week"] == self.current_week]
          
          def make_matrix(column):
               matrix = current_data.pivot_table(
                    index="store_id",columns="drug_id",values=column,aggfunc="last")

               matrix = matrix.reindex(
                    index=self.supply_data.store_id,
                    columns=self.supply_data.drug_ids,
                    fill_value=0
               )

               return matrix.fillna(0).to_numpy(dtype=np.float32)

          observation = {
               "stock_on_hand": make_matrix("stock_on_hand"),
               "forecast_demand": make_matrix("forecast_demand"),
               "stockout_history": make_matrix("stockout_history"),
               "unmet_demand_history": make_matrix("unmet_demand_history"),
               "inventory_coverage": make_matrix("inventory_coverage"),

               "stockout_risk": make_matrix("stockout_risk").astype(np.int32),

               "investment_capital": np.array(
                    [self.investment_capital],
                    dtype=np.float32
                    ),
                "disease_demand": self.supply_data.disease_pressure[:, :, self.current_week].astype(np.float32),
                "wearable_load":np.asarray(self.supply_data.wearable_signal[:,self.current_week].astype(np.float32)
                                 if self.use_wearable else np.zeros(self.n_stores, dtype=np.float32)).reshape(-1)

     }

          return observation
     
     def apply_investment(self, investment_actions):
       
        investment_actions = np.asarray(
        investment_actions, dtype=np.float32
    )

        if investment_actions.shape != (4,):
            raise ValueError("Expected four investment actions.")

        investment_actions = np.clip(investment_actions, 0.0, 1.0)

        # The four actions represent fractions of remaining capital.
        total_action = float(investment_actions.sum())

        # Prevent investing more than the available capital.
        if total_action > 1.0:
            investment_actions = investment_actions / total_action

        available_capital = self.investment_capital

        investment_results = {}
        total_invested = 0.0

        for index, area in enumerate(self.investment_areas):

            investment_amount = (
                available_capital * float(investment_actions[index])
            )

            self.investment_by_area[area] += investment_amount

            investment_results[area] = investment_amount
            total_invested += investment_amount

        self.invested_capital += total_invested
        self.investment_capital -= total_invested
        self.investment_capital = max(
    0.0,
    self.investment_capital
)

        return {
    "investment": investment_results,
    "investment_by_area": self.investment_by_area.copy(),
    "total_invested": total_invested,
    "cumulative_invested": self.invested_capital,
    "capital_remaining": self.investment_capital
}

     
     def reset(self, seed=None, options=None):
          super().reset(seed=seed)
          

          self.current_week = 0
          
          self.investment_capital = self.initial_balance
          self.invested_capital = 0.0

          self.investment_by_area = {
               "forecasting": 0.0,
               "inventory_management": 0.0,
               "distribution": 0.0,
               "supply_chain_capacity": 0.0
           }
          self.technology_capabilities = {
      area: 0.0 for area in self.investment_areas
  }
          self.previous_stockout_rate = 0.0
          self.previous_unmet_demand = 0.0
          self.previous_service_level = 0.0
          self.inventory = self.supply_data.stock[:, :, 0].copy()
          
          
          self.pending_orders = []

          self.store_drug_pairs = [
        (s, d)
        for s in range(self.supply_data.n_stores)
        for d in range(self.supply_data.n_drugs)
    ]
          self.pair_index = 0
          self.store_idx, self.drug_idx = self.store_drug_pairs[0]
          

          observation = self.get_observation()

          return observation, {}

      
     def inventory_transition(
               self,
               current_week,
               store_idx,
               drug_idx,
               current_inventory,
               next_week_demand,
               forecast_demand,
               normal_lead_time,
               unit_cost,
               action,
               pending_orders,
               state
               ):

          next_week = current_week + 1

          expedited = (action.get("expedited", False)and allow_expedite(state))

          # 3. Calculate the required order quantity
          order_action = float(np.clip(action["order"], 0.0, 1.0))

          order_quantity = int(
    np.ceil(order_action * forecast_demand * 2)
)
          # 4. Calculate procurement cost
          if order_quantity > 0 and unit_cost <= 0:
               raise ValueError("Unit procurement cost must be positive.")
          cost_per_unit = unit_cost

          if expedited:
               cost_per_unit *= (1 + EXPEDITE_COST)

          order_cost = order_quantity * cost_per_unit



          # 6. Calculate delivery time
          delivery_time = calculate_delivery_time(
               normal_lead_time,
               expedited
          )

          # 7. Schedule the supplier order
          if order_quantity > 0:

               pending_orders.append({
                    "store_idx": store_idx,
                    "drug_idx": drug_idx,
                    "quantity": order_quantity,
                    "due_week": current_week + delivery_time,
                    "expedited": expedited
               })

          # 8. Receive orders due this week
          arrived_quantity = 0

          for order in pending_orders.copy():

               if (
                    order["store_idx"] == store_idx
                    and order["drug_idx"] == drug_idx
                    and order["due_week"] <= next_week
               ):

                    arrived_quantity += order["quantity"]
                    pending_orders.remove(order)

          current_inventory += arrived_quantity

          # 9. Serve next week's demand
          units_dispensed = min(
               current_inventory,
               next_week_demand
          )

          unmet_demand = max(
               next_week_demand - units_dispensed,
               0
          )

          # 10. Calculate closing inventory
          closing_inventory = (
               current_inventory - units_dispensed
          )

          # 11. Identify stockout
          stockout = int(unmet_demand > 0)

          return {
               "week": next_week,
               "inventory": closing_inventory,
               "units_dispensed": units_dispensed,
               "unmet_demand": unmet_demand,
               "stockout": stockout,
               "arrived_quantity": arrived_quantity,
               "order_quantity": order_quantity,
               "order_cost": order_cost,
               "pending_orders": pending_orders,
               "delivery_time": delivery_time
          }
     
     def step(self, action):
        next_week = self.current_week + 1

        if next_week >= self.supply_data.n_weeks:
          return self.get_observation(), 0.0, True, False, {}

        # Convert the continuous PPO action into its two components.
        action = np.asarray(action, dtype=np.float32)

        expected_size = (
            4 + self.n_drugs + (self.n_drugs * self.n_stores)
        )

        if action.shape != (expected_size,):
            raise ValueError(
                f"Expected action shape ({expected_size},), "
                f"received {action.shape}"
            )

        # 1. Four investment actions
        investment_actions = np.clip(
            action[:4], 0.0, 1.0
        )

        # 2. Existing medicine ordering actions
        order_actions = np.clip(
            action[4:4 + self.n_drugs], 0.0, 1.0
        )

        # 3. Existing store allocation actions
        allocation_actions = action[4 + self.n_drugs:].reshape(
            self.n_drugs, self.n_stores
        )

        allocation_actions = np.clip(
            allocation_actions, 0.0, 1.0
        )

        # Normalize allocation across the complete supply chain.
        total_allocation = allocation_actions.sum()

        if total_allocation > 0:
            allocation_weights = (
                allocation_actions / total_allocation
            )
        else:
            allocation_weights = np.zeros_like(allocation_actions)


        total_reward = 0.0
        investment_reward = 0.0
        # Calculate current supply-chain performance.
        total_demand = 0.0
        total_unmet_demand = 0.0
        total_stockouts = 0
        total_units_dispensed = 0.0
        investment_result = self.apply_investment(
            investment_actions
        )
        weekly_info = {
            "pair_results": [],
            "total_order_quantity": 0,
            "units_dispensed": 0.0,
            "unmet_demand": 0.0,
            "stockouts": 0,
            "order_cost": 0.0,
            "capital_remaining": 0.0,
            "weekly_reward": 0.0
        }
        

        # Process EVERY store-drug pair during this weekly action.
        for store_idx, drug_idx in self.store_drug_pairs:

            self.store_idx = store_idx
            self.drug_idx = drug_idx

            state = self.get_discrete_state(
                store_idx, drug_idx
            )

            forecast_demand = self.get_forecast(
                store_idx, drug_idx
            )

            current_order_action = float(
                order_actions[drug_idx]
            )

            desired_order_quantity = int(
                np.ceil(current_order_action * forecast_demand * 2)
            )

            expedited = (
                allow_expedite(state)
                and desired_order_quantity > 0
            )

            pair_action = {
                "order": current_order_action,
                "expedited": expedited
            }

            next_week_demand = float(
                self.supply_data.demand[
                    store_idx, drug_idx, next_week
                ]
            )

            current_inventory = float(
                self.inventory[store_idx, drug_idx]
            )

            normal_lead_time = self.supply_data.lead_time[
                store_idx
            ]

            unit_cost = float(
                self.supply_data.cost[
                    store_idx, drug_idx, self.current_week
                ]
            )

            previous_stockout_risk = self.get_stockout_risk(
                store_idx, drug_idx
            )

            previous_inventory_coverage = self.get_inventory_coverage(
                store_idx, drug_idx
            )
            pair_start = time.perf_counter()
            result = self.inventory_transition(
                current_week=self.current_week,
                store_idx=store_idx,
                drug_idx=drug_idx,
                current_inventory=current_inventory,
                next_week_demand=next_week_demand,
                forecast_demand=forecast_demand,
                normal_lead_time=normal_lead_time,
                unit_cost=unit_cost,
                action=pair_action,
                pending_orders=self.pending_orders,
                state=state
            )
            total_demand += next_week_demand

            total_unmet_demand += result["unmet_demand"]

            total_stockouts += int(result["stockout"])

            total_units_dispensed += result["units_dispensed"]
            pair_time = time.perf_counter() - pair_start
            if pair_time > 0.1:
                 print(
                      f"Slow pair: store={store_idx}, "
                      f"drug={drug_idx}, time={pair_time:.4f}s"
                      )

            # Update inventory, pending orders and actual capital.
            self.inventory[store_idx, drug_idx] = result["inventory"]

            self.pending_orders = result["pending_orders"]

            current_inventory_coverage = self.get_inventory_coverage(
                store_idx, drug_idx
            )

            current_stockout_risk = self.get_stockout_risk(
                store_idx, drug_idx
            )

            reward = calculate_reward(
                demand=next_week_demand,
                units_dispensed=result["units_dispensed"],
                unmet_demand=result["unmet_demand"],
                order_cost=result["order_cost"],
                weekly_budget=max(unit_cost * result["order_quantity"], 1.0),
                previous_stockout_risk=previous_stockout_risk,
                current_stockout_risk=current_stockout_risk,
                previous_inventory_coverage=previous_inventory_coverage,
                current_inventory_coverage=current_inventory_coverage,
                expedited=expedited
            )

            total_reward += reward

            weekly_info["total_order_quantity"] += (
                result["order_quantity"]
            )

            weekly_info["units_dispensed"] += (
                result["units_dispensed"]
            )

            weekly_info["unmet_demand"] += (
                result["unmet_demand"]
            )

            weekly_info["stockouts"] += int(
                result["stockout"]
            )

            weekly_info["order_cost"] += (
                result["order_cost"]
            )

            weekly_info["pair_results"].append({
                "store_idx": store_idx,
                "drug_idx": drug_idx,
                "order_action": current_order_action,
                "allocation_action": allocation_weights[
                    drug_idx, store_idx
                ],
                "desired_order_quantity": desired_order_quantity,
                "order_quantity": result["order_quantity"],
                "units_dispensed": result["units_dispensed"],
                "unmet_demand": result["unmet_demand"],
                "stockout": result["stockout"],
                "order_cost": result["order_cost"],
                "expedited": expedited,
                "reward": reward
            })

        # Advance the simulation ONCE after all pairs are processed.
        # Calculate current performance indicators.
        current_stockout_rate = (
            total_stockouts / max(len(self.store_drug_pairs), 1)
        )

        current_unmet_demand = (
            total_unmet_demand / max(total_demand, 1.0)
        )

        current_service_level = (
            total_units_dispensed / max(total_demand, 1.0)
        )

        # Calculate investment reward.
        investment_reward = calculate_investment_reward(
            previous_stockout_rate=self.previous_stockout_rate,
            current_stockout_rate=current_stockout_rate,
            previous_unmet_demand=self.previous_unmet_demand,
            current_unmet_demand=current_unmet_demand,
            previous_service_level=self.previous_service_level,
            current_service_level=current_service_level,
            investment_amount=investment_result["total_invested"],
            remaining_capital=self.investment_capital,
            initial_capital=self.initial_balance
        )
        weekly_info["investment_reward"] = investment_reward
        weekly_info["stockout_rate"] = current_stockout_rate
        weekly_info["unmet_demand_rate"] = current_unmet_demand
        weekly_info["service_level"] = current_service_level

        # Add investment reward to operational reward.
        total_reward += investment_reward

        # Save current performance for the next decision.
        self.previous_stockout_rate = current_stockout_rate
        self.previous_unmet_demand = current_unmet_demand
        self.previous_service_level = current_service_level
        self.current_week = next_week

        self.pair_index = 0

        terminated = (
            self.current_week >= self.supply_data.n_weeks - 1
        )

        truncated = False

        if not terminated:
            self.store_idx, self.drug_idx = (
                self.store_drug_pairs[0]
            )
        

        observation = self.get_observation()
        weekly_info.update({
    "week": self.current_week,
    "initial_capital": self.initial_balance,
    "total_invested":investment_result["total_invested"],
    "capital_remaining": self.investment_capital,

    # Investment made during this step
    "investment": investment_result["investment"],
    "investment_by_area": self.investment_by_area.copy(),

    # Cumulative investment across all steps
    "cumulative_invested": investment_result["cumulative_invested"],

    # Performance measurements
    "investment_reward": investment_reward,
    "stockout_rate": current_stockout_rate,
    "unmet_demand_rate": current_unmet_demand,
    "service_level": current_service_level,
    "weekly_reward": total_reward
})
       
        return (
            observation,
            float(total_reward),
            terminated,
            truncated,
            weekly_info
        )
     

     
     def get_forecast(self, store_idx, drug_idx):
          return float(
        self.forecast_cube[
            store_idx,
            drug_idx,
            self.current_week
        ]
    )
     def get_inventory_coverage(self, store_idx, drug_idx):
          inventory = self.inventory[store_idx, drug_idx]

          forecast = self.get_forecast(
               store_idx, drug_idx
          )

          lead_time = self.supply_data.lead_time[store_idx]

          lead_time_demand = max(
               forecast * lead_time, 1
          )

          return float(inventory / lead_time_demand)
     def get_stockout_risk(self, store_idx, drug_idx):
          coverage = self.get_inventory_coverage(
               store_idx, drug_idx
               )

          if coverage < 0.5:
               return 3

          elif coverage < 1.0:
               return 2

          elif coverage < 1.5:
               return 1

          return 0

     def get_discrete_state(self, store_idx, drug_idx):
         # Inventory coverage
         coverage = self.get_inventory_coverage(
              store_idx, drug_idx
    )

         if coverage < 0.5:
               inventory_state = 0
         elif coverage < 1.0:
               inventory_state = 1
         elif coverage < 1.5:
               inventory_state = 2
         else:
              inventory_state = 3

          # Demand trend
         forecast = self.get_forecast(
               store_idx, drug_idx
          )

         baseline = self.supply_data.baseline[drug_idx]

         if forecast > baseline * 1.1:
               demand_state = 2
         elif forecast < baseline * 0.9:
               demand_state = 0
         else:
               demand_state = 1

          # Historical stockout
         store_id = self.supply_data.store_id[store_idx]
         drug_id = self.supply_data.drug_ids[drug_idx]
       
         history = 0

         try:
             row = self.stockout_lookup.loc[
                 (store_id, drug_id, self.current_week)
             ]

             if not row.empty:
                 history = row["stockout_history"]

         except KeyError:
             history = 0

         if pd.isna(history):
              history = 0

         if history < 0.1:
               stockout_state = 0
         elif history < 0.4:
               stockout_state = 1
         else:
               stockout_state = 2

          # Supplier lead-time risk
         lead_time = self.supply_data.lead_time[store_idx]

         median_lead_time = np.median(
               self.supply_data.lead_time
          )

         lead_time_state = int(
               lead_time > median_lead_time
          )

          # Capital availability
         capital_ratio = (
              self.investment_capital / max(self.initial_balance, 1)
         )

         if capital_ratio >= 0.5:
               capital_state = 2
         elif capital_ratio >= 0.2:
               capital_state = 1
         else:
               capital_state = 0

         return (
              inventory_state,
              demand_state,
              stockout_state,
              lead_time_state,
              capital_state
         )

     


        
          
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

EPISODES = 10
def calculate_reward(
    demand,
    units_dispensed,
    unmet_demand,
    order_cost,
    weekly_budget,
    previous_stockout_risk,
    current_stockout_risk,
    previous_inventory_coverage,
    current_inventory_coverage,
    expedited
):

    service_level = units_dispensed / max(demand, 1)

    stockout_improvement = (
        previous_stockout_risk - current_stockout_risk
    )

    coverage_improvement = (
        current_inventory_coverage - previous_inventory_coverage
    )

    cost_ratio = order_cost / max(weekly_budget, 1)

    unmet_ratio = unmet_demand / max(demand, 1)

    reward = (
        10.0 * service_level
        + 5.0 * stockout_improvement
        + 2.0 * np.clip(coverage_improvement, -1, 1)
        - 10.0 * unmet_ratio
        - 3.0 * cost_ratio
    )

    if expedited and current_stockout_risk == 0:
        reward -= 2.0

    return float(reward)

def calculate_investment_reward(
    previous_stockout_rate,
    current_stockout_rate,
    previous_unmet_demand,
    current_unmet_demand,
    previous_service_level,
    current_service_level,
    investment_amount,
    remaining_capital,
    initial_capital
):
    """
    Evaluate the supply-chain performance of technology investments.
    """

    # Improvement in stockout performance.
    stockout_improvement = (
        previous_stockout_rate - current_stockout_rate
    )

    # Improvement in unmet demand.
    unmet_demand_improvement = (
        previous_unmet_demand - current_unmet_demand
    )

    # Improvement in service level.
    service_improvement = (
        current_service_level - previous_service_level
    )

    # Normalize the investment expenditure.
    investment_ratio = (
        investment_amount / max(initial_capital, 1.0)
    )

    # Capital preservation.
    capital_ratio = (
        remaining_capital / max(initial_capital, 1.0)
    )

    # Reward actual supply-chain improvement.
    performance_reward = (
        10.0 * stockout_improvement
        + 5.0 * unmet_demand_improvement
        + 10.0 * service_improvement
    )

    # Penalize expenditure without corresponding improvement.
    investment_penalty = 2.0 * investment_ratio

    # Reward retaining unused capital.
    capital_reward = 0.5 * capital_ratio

    investment_reward = (
        performance_reward
        - investment_penalty
        + capital_reward
    )

    return float(investment_reward)

def allow_expedite(state):

    inventory_state = state[0]
    demand_state = state[1]
    stockout_state = state[2]
    lead_time_state = state[3]

    return (
        inventory_state <= 1 and
        (demand_state == 2 or stockout_state == 2) and
        lead_time_state == 1
    )
def get_state(row, capital, initial_balance, median_lead_time):

    # 1. Inventory coverage
    if row["inventory_coverage"] < 0.5:
        inventory_state = 0
    elif row["inventory_coverage"] < 1.0:
        inventory_state = 1
    elif row["inventory_coverage"] < 1.5:
        inventory_state = 2
    else:
        inventory_state = 3

    # 2. Demand trend
    demand_state = int(row["demand_trend"])

    # 3. Historical stockout frequency
    if row["stockout_history"] < 0.1:
        stockout_state = 0
    elif row["stockout_history"] < 0.3:
        stockout_state = 1
    else:
        stockout_state = 2

    # 4. Supplier lead-time risk
    lead_time_state = int(
        row["supplier_lead_time_weeks"] > median_lead_time
    )

    # 5. Capital availability
    capital_ratio = capital / initial_balance

    if capital_ratio < 0.25:
        capital_state = 0
    elif capital_ratio < 0.60:
        capital_state = 1
    else:
        capital_state = 2

    return (
        inventory_state,
        demand_state,
        stockout_state,
        lead_time_state,
        capital_state
    )

STATE_SIZE =  (4, 3, 3, 2, 3)

ACTIONS = {
    0: "hold",
    1: "small_order",
    2: "medium_order",
    3: "large_order",
    4: "expedite_small",
    5: "expedite_medium",
    6: "expedite_large"
}

N_ACTIONS = len(ACTIONS)


Q_TABLE = np.zeros(
    (np.prod(STATE_SIZE), N_ACTIONS),
    dtype=np.float32
)
def update_q_table(
    state,
    action,
    reward,
    next_state,
    terminated
):

    state_index = np.ravel_multi_index(
        state, STATE_SIZE
    )

    next_state_index = np.ravel_multi_index(
        next_state, STATE_SIZE
    )

    current_q = Q_TABLE[state_index, action]

    if terminated:

        target = reward

    else:

        valid_actions = get_valid_actions(next_state)

        next_q = np.max(
            Q_TABLE[next_state_index, valid_actions]
        )

        target = reward + DISCOUNT_FACTOR * next_q

    Q_TABLE[state_index, action] = (current_q
        + LEARNING_RATE * (target - current_q)
    )

def get_order_quantity(action, forecast_demand):

    if action in [0]:
        return 0

    if action in [1, 4]:
        return forecast_demand * 0.5

    if action in [2, 5]:
        return forecast_demand * 1.0

    if action in [3, 6]:
        return forecast_demand * 2.0
def is_expedited(action):

    return action in [4, 5, 6]

def calculate_delivery_time(normal_lead_time, expedite):

    if expedite:

        delivery_time = max(
            MIN_EXPEDITE_LEAD_TIME,
            normal_lead_time - EXPEDITE_REDUCTION
        )

    else:
        delivery_time = normal_lead_time

    return delivery_time

def get_valid_actions(state):

    valid_actions = [0, 1, 2, 3]

    if allow_expedite(state):

        valid_actions.extend([4, 5, 6])

    return valid_actions
def choose_action(state, epsilon):

    state_index = np.ravel_multi_index(
        state, STATE_SIZE
    )

    valid_actions = get_valid_actions(state)

    if np.random.random() < epsilon:
        return int(np.random.choice(valid_actions))

    valid_q_values = Q_TABLE[state_index, valid_actions]

    return int(
        valid_actions[np.argmax(valid_q_values)]
    )
class DQNStrategyWrapper(gymnasium.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.action_space = gymnasium.spaces.Box(
    low=0.0,
    high=1.0,
    shape=(4,),
    dtype=np.float32
)

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                gymnasium.spaces.flatten_space(
                    env.observation_space
                ).shape
            ),
            dtype=np.float32
        )
        self.action_intensity = {
            0: 0.00,
            1: 0.25,
            2: 0.50,
            3: 1.00,
            4: 0.25,
            5: 0.50,
            6: 1.00
        }

    def observation(self, obs):
        flattened_obs = gymnasium.spaces.flatten(
            self.env.observation_space,obs
    )

        return np.asarray(flattened_obs,dtype=np.float32)
   
        
        
        
    def step(self, investment_actions):
        global EPSILON

         # Get the current observation from the environment
        current_obs = self.env.unwrapped.get_observation()
        
        base_env = self.env.unwrapped

        investment_actions = np.asarray(
            investment_actions,
            dtype=np.float32
        ).reshape(-1)
        investment_actions = np.clip(
        investment_actions,
        0.0,
        1.0
    )

        if investment_actions.shape != (4,):
            raise ValueError(
                f"Expected four investment actions, "
                f"received {investment_actions.shape}"
            )

        investment_actions = np.clip(
            investment_actions,
            0.0,
            1.0
        )

        n_stores = base_env.n_stores
        n_drugs = base_env.n_drugs

        store_drug_pairs = list(
            base_env.store_drug_pairs
        )

        # Save the current Q-learning states.
        pair_states = {}

        for store_idx, drug_idx in store_drug_pairs:

            state = tuple(
                base_env.get_discrete_state(
                    store_idx,
                    drug_idx
                )
            )

            pair_states[(store_idx, drug_idx)] = state

        # Generate operational actions using Q-learning.
        pair_actions = {}

        order_actions = np.zeros(
            n_drugs,
            dtype=np.float32
        )

        allocation_actions = np.zeros(
            (n_drugs, n_stores),
            dtype=np.float32
        )

        for store_idx, drug_idx in store_drug_pairs:

            pair = (store_idx, drug_idx)

            state = pair_states[pair]

            q_action = choose_action(
                state,
                EPSILON
            )

            pair_actions[pair] = q_action

            intensity = self.action_intensity[q_action]

            order_actions[drug_idx] = max(
                order_actions[drug_idx],
                intensity
            )

            forecast = base_env.get_forecast(
                store_idx,
                drug_idx
            )

            allocation_actions[
                drug_idx,
                store_idx
            ] = intensity * max(
                float(forecast),
                0.0
            )

        # Combine PPO investment decisions and
        # Q-learning operational decisions.
        combined_action = np.concatenate([
            investment_actions,
            order_actions,
            allocation_actions.flatten()
        ]).astype(np.float32)

        # Execute the combined action.
        next_obs, reward, terminated, truncated, info = (
            self.env.step(combined_action)
        )

        # Update the Q-table using individual pair rewards.
        pair_results = info.get("pair_results")

        if pair_results is None:
            raise KeyError(
                "Missing pair_results in environment info."
            )

        if len(pair_results) != len(store_drug_pairs):
            raise ValueError(
                "pair_results does not match store-drug pairs."
            )

        for (store_idx, drug_idx), result in zip(
            store_drug_pairs,
            pair_results
        ):

            pair = (store_idx, drug_idx)

            state = pair_states[pair]
            q_action = pair_actions[pair]

            pair_reward = float(
                result["reward"]
            )

            if terminated or truncated:

                next_state = state

            else:

                next_state = tuple(
                    base_env.get_discrete_state(
                        store_idx,
                        drug_idx
                    )
                )

            update_q_table(
                state=state,
                action=q_action,
                reward=pair_reward,
                next_state=next_state,
                terminated=terminated or truncated
            )

        # Reduce Q-learning exploration at episode boundaries.
        if terminated or truncated:

            EPSILON = max(
                EPSILON_MIN,
                EPSILON * EPSILON_DECAY
            )

        # Flatten the next observation for PPO.
        flattened_obs = gymnasium.spaces.flatten(
            self.env.observation_space,
            next_obs
        )

        
        return (
            np.asarray(flattened_obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info
        )
        
    


    def reset(self,  seed=None, options=None):

        obs, info = self.env.reset( seed=None, options=None)
        flattened_obs = self.observation(obs)


        return np.asarray(flattened_obs,dtype=np.float32), info
import json
def train_agent(env, episodes=EPISODES):

    global EPSILON

    episode_rewards = []
    base_env = env.unwrapped

    n_stores = base_env.n_stores
    n_drugs = base_env.n_drugs

    store_drug_pairs = list(base_env.store_drug_pairs)
    action_intensity = {
        0: 0.00,
        1: 0.25,
        2: 0.50,
        3: 1.00,
        4: 0.25,
        5: 0.50,
        6: 1.00
    }
    investment_history = []


    for episode in range(episodes):

        obs, info = env.reset()
  
        total_reward = 0.0

        terminated = False
        truncated = False

        while not terminated or truncated:

            pair_states = {}

            for store_idx, drug_idx in store_drug_pairs:

                state = tuple(
                    base_env.get_discrete_state(
                        store_idx,
                        drug_idx
                    )
                )

                pair_states[(store_idx, drug_idx)] = state

            order_actions = np.zeros(
                n_drugs,
                dtype=np.float32
            )

            allocation_actions = np.zeros(
                (n_drugs, n_stores),
                dtype=np.float32
            )

            # 2. Choose one Q-learning action per pair
            pair_actions = {}

            
            for store_idx, drug_idx in store_drug_pairs:

                pair = (store_idx, drug_idx)

                state = pair_states[pair]

                action = choose_action(state, EPSILON)

                pair_actions[pair] = action

            
            order_actions = np.zeros(
                n_drugs,
                dtype=np.float32
            )

            allocation_actions = np.zeros(
                (n_drugs, n_stores),
                dtype=np.float32
            )

            for (store_idx, drug_idx), action in pair_actions.items():

                intensity = action_intensity[action]

                # The environment shares one order intensity
                # across all stores for each drug.
                order_actions[drug_idx] = max(
                    order_actions[drug_idx],
                    intensity
                )

                # Allocate more capital to pairs requesting
                # larger orders and having greater demand.
                forecast = base_env.get_forecast(
                    store_idx,
                    drug_idx
                )

                allocation_actions[
                    drug_idx,
                    store_idx
                ] = intensity * max(float(forecast), 0.0)

                investment_actions = np.random.uniform(
    0.0, 1.0, size=4
).astype(np.float32)

            # Combine both action components.
            continuous_action = np.concatenate([
    investment_actions,          # 4 investment actions
    order_actions,               # 19 ordering actions
    allocation_actions.flatten() # 304 allocation actions
]).astype(np.float32)


            next_obs, reward, terminated, truncated, info = (env.step(continuous_action))
            investment_history.append({
    "week": info["week"],
    "investment": info["investment"],
    "investment_by_area": info["investment_by_area"],
    "total_invested": info["total_invested"],
    "cumulative_invested": info["cumulative_invested"],
    "capital_remaining": info["capital_remaining"],
    "investment_reward": info["investment_reward"],
    "stockout_rate": info["stockout_rate"],
    "unmet_demand_rate": info["unmet_demand_rate"],
    "service_level": info["service_level"]
})
            

            total_reward += float(reward)
            pair_results = info.get("pair_results")

            if pair_results is None:
                raise KeyError(
                    "The environment did not return "
                    "'pair_results' in its info dictionary. "
                    "The Q-table needs an individual reward "
                    "for each store-drug pair."
                )

            if len(pair_results) != len(store_drug_pairs):
                raise ValueError(
                    "The number of pair_results does not match "
                    "the number of store-drug pairs.")
            

            with open("investment_history.json", "w") as file:
                json.dump(investment_history, file, indent=4)

            
            for (store_idx, drug_idx), result in zip(
                store_drug_pairs,
                pair_results
            ):

                state = pair_states[
                    (store_idx, drug_idx)
                ]

                pair = (store_idx, drug_idx)

                if pair not in pair_actions:
                     raise KeyError(
                          f"Missing action for pair {pair}. "
                          f"Available keys: {list(pair_actions.keys())[:10]}"
    )

                action = pair_actions[pair]
                pair_reward = float(result["reward"])

                # At the end of an episode, there is no
                # valid next-week state.
                if terminated or truncated:

                    next_state = state

                else:

                    next_state = tuple(
                        base_env.get_discrete_state(
                            store_idx,
                            drug_idx
                        )
                    )


            # Update Q-table
            update_q_table(
                state=tuple(state),
                action=action,
                reward=pair_reward,
                next_state=tuple(next_state),
                terminated=terminated or truncated
            )
        obs = next_obs
            
        # Reduce exploration
        EPSILON = max(
            EPSILON_MIN,
            EPSILON * EPSILON_DECAY
        )

        episode_rewards.append(total_reward)

        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Epsilon: {EPSILON:.4f} | "
            f"Q-table updated: {np.count_nonzero(Q_TABLE)}"
                f"Episode: {episode + 1}, "
                f"total_reward: {total_reward:.2f}, "
                f"Epsilon: {EPSILON:.4f}"
            )

    return episode_rewards

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

weekly_disease_demand = (df5.groupby(["week", "disease_category"]).size().reset_index(name="disease_cases"))


data = {
    "ehr": df,
    "clinical": df5,
    "context": df4,
    "iot": df7
}
gymnasium.register(id="DigitalHealth-v0", entry_point=DigitalHealthEnv, kwargs={
        "data": data,
        "supply_data": supply_data
    }
)
env = gymnasium.make("DigitalHealth-v0")
state = env.unwrapped.get_discrete_state(
    env.unwrapped.store_idx,
    env.unwrapped.drug_idx
)
#train_agent(env)

dqn_env = DQNStrategyWrapper(env)
training_env = gymnasium.wrappers.FlattenObservation(env)

obs, info = dqn_env.reset()
print("Environment type:", type(dqn_env))
print("Observation type:", type(obs))
print("Observation shape:", obs.shape)
print("Observation shape:",
      obs.shape)

print("DQN action space:",
      dqn_env.action_space)


# start_time = time.perf_counter()

# action = env.action_space.sample()

# obs, reward, terminated, truncated, info = env.step(action)

# print("Weekly execution time:",
#       time.perf_counter() - start_time)

# print("Pairs processed:", len(info["pair_results"]))

# train_agent(env)
# # Check Q-table after training
# print("\n========== Q-LEARNING RESULTS ==========")

# print("Q-table:")
# print(Q_TABLE)

# print("Non-zero Q-values:", np.count_nonzero(Q_TABLE))
# print("Maximum Q-value:", np.max(Q_TABLE))
# print("Minimum Q-value:", np.min(Q_TABLE))
# print("STATE:", state)
# print("STATE TYPE:", type(state))
# print("STATE SIZES:", STATE_SIZE)
print("Observation space:", dqn_env.observation_space)
print("Action space:", dqn_env.action_space)
raw_obs = env.unwrapped.get_observation()

print("\n========== ORIGINAL OBSERVATION DIAGNOSTICS ==========")

for key, value in raw_obs.items():
    values = np.asarray(value, dtype=np.float32)

    print(
        f"{key}: "
        f"shape={values.shape}, "
        f"min={np.nanmin(values):.3f}, "
        f"max={np.nanmax(values):.3f}, "
        f"mean={np.nanmean(values):.3f}"
    )
model = PPO(
    "MlpPolicy",
    dqn_env,
    learning_rate=0.0003,
    n_steps=100,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
print("PPO action space:", model.action_space)
print("Wrapper action space:", dqn_env.action_space)
print("Starting PPO training...")

# obs, info = env.reset()

# print("Environment reset successful.")

# action = env.action_space.sample()

# obs, reward, terminated, truncated, info = env.step(action)

# print("Environment step successful.")
# print("Reward:", reward)

model.learn(
    total_timesteps=500,
    log_interval=1
)

print("PPO training completed.")
model.save('ppo_investment')
model = PPO.load("deployment/ppo_investment")
def deploy(data,observation_space):
        observation = flatten(
            observation_space,
            data
        ).astype(np.float32)
        
        if observation.shape != (2145,):
            raise ValueError(F"Expected 2145 features, got{observation.shape}")
        action, _ = model.predict(observation, deterministic=True)
        print("PPO action space:", model.action_space)
        print("PPO raw prediction:", action)
        print("PPO prediction shape:", np.asarray(action).shape)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != 4:
            raise ValueError(
                f"PPO model has action space {model.action_space}, "
                f"but prediction returned {action.size} values: {action}"
            )

        return action
# Check Q-table after training
# print("\n========== Q-LEARNING RESULTS ==========")

# print("Q-table:")
# print(Q_TABLE)

# print("Non-zero Q-values:", np.count_nonzero(Q_TABLE))
print("Maximum Q-value:", np.max(Q_TABLE))
print("Minimum Q-value:", np.min(Q_TABLE))
print("Q-table changes:", np.count_nonzero(Q_TABLE))
print("Q-table maximum:", np.max(Q_TABLE))
print("Current epsilon:", EPSILON)