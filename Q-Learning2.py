import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
PATHS = {   # every path must end with a .csv file name
    "drugs":      "C:/Users/VP/Downloads/archive SCD/drugs.csv",
    "stores":     "C:/Users/VP/Downloads/archive SCD/stores.csv",
    "dispensing": "C:/Users/VP/Downloads/archive SCD/dispensing.csv",
    "context":    "C:/Users/VP/Downloads/archive SCD/context.csv",
    "visits":     "C:/Users/VP/Downloads/archive (4)/visits_encounters.csv",
    "clinical":   "C:/Users/VP/Downloads/archive (4)/clinical_details.csv",
    "wearable":   "C:/Users/VP/Downloads/archive IoTBWD/AIoT_Orienteering_Dataset/AIoT_Orienteering_Full_Dataset.csv",
}

SIGNALS = ["latitude", "longitude", "altitude", "speed", "acc_x", "acc_y", "acc_z",
           "gyro_x", "gyro_y", "gyro_z", "heart_rate"]
DISEASES = ["dengue", "malaria", "flu", "diarrhoeal",
            "respiratory", "typhoid", "chikungunya", "leptospirosis"]
START = pd.Timestamp("2023-01-02")       # week 0 in dispensing.csv

EXPEDITE_COST = 0.20                     # expedited orders cost 20% more per unit
EXPEDITE_REDUCTION = 2                   # and arrive 2 weeks sooner...
MIN_EXPEDITE_LEAD_TIME = 1               # ...but never faster than 1 week
COST_RATIO = 0.70                        # ASSUMPTION: purchase cost = 70% of selling price

# which EHR disease category pushes demand for which drug (my judgment, please review)
CATEGORY_TO_DRUGS = {
    "Respiratory":      ["D10", "D11", "D12", "D14", "D19"],
    "Infectious":       ["D08", "D11", "D12", "D16"],
    "Cardiovascular":   ["D03", "D04", "D05"],
    "Endocrine":        ["D01", "D02", "D07"],
    "Gastrointestinal": ["D09", "D17", "D18"],
}
OUTBREAK_CUTS = [0.005, 0.1]             # measured from the data
HOSPITAL_CUTS = [0.16, 0.32]


# ----------------------------------------------------------------------
# 2. DATA LAYER
# ----------------------------------------------------------------------
class SupplyData:

    def __init__(self, paths=None, hr_high=100.0, assign_wearables=True, seed=0,
                 min_ehr_admissions=20):
        super().__init__()
        paths = paths or PATHS

        # ---- load the files ----
        df1 = pd.read_csv(paths["drugs"])
        df2 = pd.read_csv(paths["stores"])
        df3 = pd.read_csv(paths["dispensing"])
        df4 = pd.read_csv(paths["context"])
        visitor = pd.read_csv(paths["visits"], parse_dates=["admission_datetime"])
        clini_detail = pd.read_csv(paths["clinical"])
        df5 = visitor.merge(clini_detail[["encounter_id", "disease_category", "clinical_record_id",
                                          "disease_severity", "primary_diagnosis_name"]],
                            on="encounter_id")
        df6 = pd.read_csv(paths["wearable"])
        df7 = df6.sort_values(["participant_id", "session_id", "timestamp"])

        # ---- pharmacy basics (must come BEFORE build_stockout_features, which uses self.stores) ----
        self.drugs, self.stores = df1, df2
        self.drug_ids = df1["drug_id"].tolist()
        self.store_id = df2["store_id"].tolist()
        self.n_drugs, self.n_stores = len(df1), len(df2)
        self.n_weeks = int(df3["week"].max()) + 1

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

        # ---- stockout features (your pandas table), then turned into arrays for speed ----
        self.stockout_features = self.build_stockout_features(df3)
        sf = self.stockout_features
        forecast = self.table_to_cube(sf, "forecast_demand")
        # fall back to the drug's baseline where there is no usable forecast
        self.forecast_cube = np.where(np.isfinite(forecast) & (forecast > 0), forecast,
                                      self.baseline[None, :, None]).astype(np.float32)
        self.stockout_history_cube = np.nan_to_num(
            self.table_to_cube(sf, "stockout_history")).astype(np.float32)

        # ---- outbreaks: store x disease x week (each store inherits its region) ----
        regions = df4["region"].unique().tolist()
        ri = {r: i for i, r in enumerate(regions)}
        sev = np.zeros((len(regions), len(DISEASES), self.n_weeks), dtype=np.float32)
        r_idx = df4["region"].map(ri).to_numpy()
        w4 = df4["week"].to_numpy()
        for k, name in enumerate(DISEASES):
            sev[r_idx, k, w4] = df4[f"sev_{name}"].to_numpy()
        self.outbreak = sev[df2["region"].map(ri).to_numpy()]

        link = np.zeros((self.n_drugs, len(DISEASES)), dtype=np.float32)
        for d, txt in enumerate(df1["responds_to"]):
            for name in str(txt).split(";"):
                if name in DISEASES:
                    link[d, DISEASES.index(name)] = 1.0
        self.drug_disease_link = link

        # ---- EHR (df5): weekly admissions per disease category ----
        df5["week"] = (df5["admission_datetime"].dt.normalize() - START).dt.days // 7
        self.df5 = df5
        inside = df5[(df5["week"] >= 0) & (df5["week"] < self.n_weeks)]
        self.categories = sorted(inside["disease_category"].unique())
        counts = (inside.groupby(["disease_category", "week"]).size().unstack(fill_value=0)
                  .reindex(index=self.categories, columns=range(self.n_weeks), fill_value=0))
        self.weekly = counts.to_numpy(dtype=np.float32)                 # (category, week)
        self.total_admissions = self.weekly.sum(axis=0)
        norm = self.weekly / np.maximum(self.weekly.max(axis=1, keepdims=True), 1)

        self.hospital_load = np.zeros((self.n_drugs, self.n_weeks), dtype=np.float32)
        for d, drug_id in enumerate(self.drug_ids):
            rows = [self.categories.index(c) for c, ds in CATEGORY_TO_DRUGS.items()
                    if drug_id in ds and c in self.categories]
            if rows:
                self.hospital_load[d] = norm[rows].mean(axis=0)
        self.ehr_valid = self.total_admissions >= min_ehr_admissions   # weeks with enough EHR volume

        # ---- MATCH: disease signals -> each store x drug x week ----
        # outbreak side: average severity of the diseases this drug responds to
        links_per_drug = np.maximum(link.sum(axis=1), 1.0)
        outbreak_cube = np.einsum("skw,dk->sdw", self.outbreak, link) / links_per_drug[None, :, None]
        # hospital side: admissions pressure of the EHR categories linked to this drug (0 if EHR invalid)
        hospital_cube = np.where(self.ehr_valid[None, :], self.hospital_load, 0.0)[None, :, :]
        hospital_cube = np.broadcast_to(hospital_cube, outbreak_cube.shape)
        # one 0/1/2 signal: "either source says demand pressure is high"
        self.disease_state = np.maximum(np.digitize(outbreak_cube, OUTBREAK_CUTS),
                                        np.digitize(hospital_cube, HOSPITAL_CUTS)).astype(np.int8)

        # ---- wearable: participant x session x second ----
        self.participant_ids = sorted(df7["participant_id"].unique().tolist())
        self.n_participants = len(self.participant_ids)
        self.n_sessions = int(df7["session_id"].max())
        self.n_seconds = int(df7["timestamp"].max()) + 1
        self.hr_high = hr_high

        pi = {p: i for i, p in enumerate(self.participant_ids)}
        p_idx = df7["participant_id"].map(pi).to_numpy()
        s_idX = df7["session_id"].to_numpy() - 1
        t_idx = df7["timestamp"].to_numpy()

        shape = (self.n_participants, self.n_sessions, self.n_seconds)
        self.signals = {}
        for name in SIGNALS:
            a = np.zeros(shape, dtype=np.float32)
            a[p_idx, s_idX, t_idx] = df7[name].to_numpy()
            self.signals[name] = a

        terrain = np.zeros(shape, dtype=np.float32)
        terrain[p_idx, s_idX, t_idx] = df7["terrain_type"].to_numpy()
        self.terrain = terrain
        self.build_session()

        # ---- link wearables to stores (ASSUMED random but reproducible assignment) ----
        rng = np.random.default_rng(seed)
        self.cohort_assignment = rng.integers(0, self.n_stores, size=self.n_participants)
        self.assign_wearables = assign_wearables
        self.cohort_load = np.zeros(self.n_stores, dtype=np.float32)
        self.cohort_hr_high = np.zeros(self.n_stores, dtype=np.float32)
        self.cohort_size = np.zeros(self.n_stores, dtype=np.int32)
        self.assign_cohorts()
        cw = np.zeros((self.n_stores, self.n_weeks), dtype=np.float32)
        for s in range (self.n_stores):
            m = np.where(self.cohort_assignment == s)[0]
            if len(m):
                per_session = self.frac_hr_high[m].mean(axis=0)
                cw[s] = per_session[np.arange(self.n_weeks) % self.n_sessions]
        cuts = np.quantile(cw, [1/3, 2/3])
        self.wearable_state = np.digitize(cw, cuts).astype(np.int8)

    # ---------- your stockout feature table ----------
    def build_stockout_features(self, df3):
        df = df3.sort_values(["store_id", "drug_id", "week"]).copy()
        # 1. demand that could not be fulfilled
        df["unmet_demand"] = (df["demand"] - df["units_dispensed"]).clip(lower=0)
        grp = df.groupby(["store_id", "drug_id"])

        def prev_avg(col, lag=1, window=4):
            # average of the `window` weeks that ended `lag` weeks ago, per store x drug
            return grp[col].transform(lambda x: x.shift(lag).rolling(window, min_periods=1).mean())

        # 2-3. recent stockout frequency and unmet demand (previous 4 weeks)
        df["stockout_history"] = prev_avg("stockout").fillna(0)
        df["unmet_demand_history"] = prev_avg("unmet_demand").fillna(0)
        # 4. forecast = average demand of the latest 4 recorded weeks
        df["forecast_demand"] = prev_avg("demand")
        previous_demand = prev_avg("demand", lag=1)
        earlier_demand = prev_avg("demand", lag=5)
        df["demand_trend"] = np.select([previous_demand > earlier_demand * 1.1,
                                        previous_demand < earlier_demand * 0.9],
                                       [2, 0], default=1)
        # 5. supplier lead time of each store
        df = df.merge(self.stores[["store_id", "supplier_lead_time_weeks"]],
                      on="store_id", how="left")
        # 6. demand expected during the supplier lead time
        df["lead_time_demand"] = df["forecast_demand"] * df["supplier_lead_time_weeks"]
        # 7. inventory coverage (using the stock recorded in the data)
        df["inventory_coverage"] = df["stock_on_hand"] / df["lead_time_demand"].clip(lower=1)
        # 8. stockout risk: 3 critical, 2 high, 1 moderate, 0 low
        df["stockout_risk"] = np.select([df["inventory_coverage"] < 0.5,
                                         df["inventory_coverage"] < 1.0,
                                         df["inventory_coverage"] < 1.5],
                                        [3, 2, 1], default=0)
        return df

    def table_to_cube(self, df, col):
        """Turn a column of the stockout table into a (store, drug, week) array."""
        a = np.full((self.n_stores, self.n_drugs, self.n_weeks), np.nan, dtype=np.float32)
        a[df["store_id"].map(self.si).to_numpy(),
          df["drug_id"].map(self.di).to_numpy(),
          df["week"].to_numpy()] = df[col].to_numpy()
        return a

    # ---------- wearables ----------
    def build_session(self):
        hr = self.signals["heart_rate"]
        acc = np.sqrt(self.signals["acc_x"] ** 2 + self.signals["acc_y"] ** 2
                      + self.signals["acc_z"] ** 2)
        gyro = np.sqrt(self.signals["gyro_x"] ** 2 + self.signals["gyro_y"] ** 2
                       + self.signals["gyro_z"] ** 2)

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

    def session(self, participant_id, session_id):
        p = self.participant_ids.index(participant_id)
        s = session_id - 1
        return dict(mean_hr=float(self.mean_hr[p, s]),
                    max_hr=float(self.max_hr[p, s]),
                    frac_hr_high=float(self.frac_hr_high[p, s]),
                    mean_speed=float(self.mean_speed[p, s]),
                    motion_intensity=float(self.motion_intensity[p, s]),
                    load_index=float(self.load_index[p, s]))

    def summary(self):
        print(f"stores={self.n_stores} drugs={self.n_drugs} pairs={self.n_stores * self.n_drugs} "
              f"weeks={self.n_weeks}")
        print(f"participants={self.n_participants} sessions={self.n_sessions} "
              f"seconds={self.n_seconds}")
        print("mean HR:", round(float(self.mean_hr.mean()), 1),
              "| heart rate threshold:", self.hr_high,
              "| share of time above it:", f"{self.frac_hr_high.mean():.1%}")
        print("EHR-valid weeks:", int(self.ehr_valid.sum()), "of", self.n_weeks)
        share = np.bincount(self.disease_state.ravel(), minlength=3) / self.disease_state.size
        print("disease signal low / medium / high:", np.round(share, 2).tolist())


# ----------------------------------------------------------------------
# 3. RULES: states, actions, rewards
# ----------------------------------------------------------------------
# state = (inventory, demand trend, stockout history, lead-time risk, capital, disease signal)
STATE_SIZE = (4, 3, 3, 2, 3, 3, 3)

ACTIONS = {
    0: "hold",
    1: "small_order",
    2: "medium_order",
    3: "large_order",
    4: "expedite_small",
    5: "expedite_medium",
    6: "expedite_large",
}
N_ACTIONS = len(ACTIONS)


def is_expedited(action):
    return action in [4, 5, 6]


def get_order_quantity(action, forecast_demand):
    if action == 0:
        return 0
    if action in [1, 4]:
        return forecast_demand * 0.5
    if action in [2, 5]:
        return forecast_demand * 1.0
    return forecast_demand * 2.0            # actions 3 and 6


def calculate_delivery_time(normal_lead_time, expedite):
    if expedite:
        return max(MIN_EXPEDITE_LEAD_TIME, normal_lead_time - EXPEDITE_REDUCTION)
    return normal_lead_time


def allow_expedite(state):
    inventory_state, demand_state, stockout_state, lead_time_state = state[0], state[1], state[2], state[3]
    return (inventory_state <= 1
            and (demand_state == 2 or stockout_state == 2)
            and lead_time_state == 1)


def get_valid_actions(state):
    valid = [0, 1, 2, 3]
    if allow_expedite(state):
        valid.extend([4, 5, 6])
    return valid


def risk_from_coverage(coverage):
    if coverage < 0.5:
        return 3
    if coverage < 1.0:
        return 2
    if coverage < 1.5:
        return 1
    return 0


def calculate_reward(demand, units_dispensed, unmet_demand, order_cost, weekly_budget,
                     previous_stockout_risk, current_stockout_risk,
                     previous_inventory_coverage, current_inventory_coverage, expedited):
    service_level = units_dispensed / max(demand, 1)
    stockout_improvement = previous_stockout_risk - current_stockout_risk
    coverage_improvement = current_inventory_coverage - previous_inventory_coverage
    cost_ratio = order_cost / max(weekly_budget, 1)
    unmet_ratio = unmet_demand / max(demand, 1)

    reward = (10.0 * service_level
              + 5.0 * stockout_improvement
              + 2.0 * np.clip(coverage_improvement, -1, 1)
              - 10.0 * unmet_ratio
              - 3.0 * cost_ratio)
    if expedited and current_stockout_risk == 0:
        reward -= 2.0                       # paid extra for speed that was not needed
    return float(reward)


# ----------------------------------------------------------------------
# 4. ENVIRONMENT: all 304 store-drug pairs share ONE capital pool
# ----------------------------------------------------------------------
class DigitalHealthEnv(gymnasium.Env):
    """
    One step = one decision for ONE store-drug pair in the current week.
    After all 304 pairs have been handled, the week moves on.
    Every order is paid from the shared balance, and every sale pays back into it.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, supply_data, initial_balance=None, start_week=0, end_week=None,
                 shuffle_pairs=False, random_start=False, min_episode_weeks=20, render_mode=None):
        super(DigitalHealthEnv,self).__init__()
        sd = supply_data
        self.supply_data = sd
        self.render_mode = render_mode
        self.n_stores, self.n_drugs = sd.n_stores, sd.n_drugs
        self.pairs = [(s, d) for s in range(sd.n_stores) for d in range(sd.n_drugs)]
        self.n_pairs = len(self.pairs)
        self.inventory = sd.stock[:, :, 0].copy()

        self.start_week = start_week
        self.end_week = end_week if end_week is not None else sd.n_weeks - 1
        self.median_lead_time = float(np.median(sd.lead_time))
        self.shuffle_pairs = shuffle_pairs
        self.random_start = random_start
        self.min_episode_weeks = min_episode_weeks

        # shared capital: pass initial_balance, or default to 4 weeks of purchasing at baseline demand
        if initial_balance is None:
            weekly_purchase = (sd.baseline[None, :] * sd.price.mean(axis=2) * COST_RATIO).sum()
            initial_balance = 4.0 * float(weekly_purchase)
        self.initial_balance = float(initial_balance)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.MultiDiscrete(list(STATE_SIZE))
        self.max_lead = int(sd.lead_time.max())
        self.reset()

    # ---------- reading the state ----------
    def _next_week_index(self):
        return min(self.current_week + 1, self.supply_data.n_weeks - 1)

    def get_forecast(self, store_idx, drug_idx):
        return float(self.supply_data.forecast_cube[store_idx, drug_idx, self._next_week_index()])

    def get_inventory_coverage(self, store_idx, drug_idx):
        # stock position = shelf stock + orders still on their way
        position = self.inventory[store_idx, drug_idx] + self.in_transit[store_idx, drug_idx]
        lead_time_demand = max(self.get_forecast(store_idx, drug_idx)
                               * self.supply_data.lead_time[store_idx], 1.0)
        return float(position / lead_time_demand)

    def get_discrete_state(self, store_idx, drug_idx):
        sd = self.supply_data

        coverage = self.get_inventory_coverage(store_idx, drug_idx)
        inventory_state = 0 if coverage < 0.5 else 1 if coverage < 1.0 else 2 if coverage < 1.5 else 3

        forecast, baseline = self.get_forecast(store_idx, drug_idx), float(sd.baseline[drug_idx])
        demand_state = 2 if forecast > baseline * 1.1 else 0 if forecast < baseline * 0.9 else 1

        history = float(sd.stockout_history_cube[store_idx, drug_idx, self._next_week_index()])
        stockout_state = 0 if history < 0.1 else 1 if history < 0.4 else 2

        lead_time_state = int(sd.lead_time[store_idx] > self.median_lead_time)

        capital_ratio = self.balance / max(self.initial_balance, 1.0)
        capital_state = 2 if capital_ratio >= 0.5 else 1 if capital_ratio >= 0.2 else 0

        disease_state = int(sd.disease_state[store_idx, drug_idx, self.current_week])

        return (inventory_state, demand_state, stockout_state,
                lead_time_state, capital_state, disease_state, int(sd.wearable_state[store_idx, self.current_week]))

    def get_observation(self):
        s, d = self.pairs[self.order[self.pair_index]]
        store_idx, drug_idx = self.pairs[self.order[self.pair_index]]
        self.state = self.get_discrete_state(s, d)
        return np.array(self.state, dtype=np.int64)
    
    def _new_order(self):
        self.order = (self.np_random.permutation(self.n_pairs) if self.shuffle_pairs
                  else np.arange(self.n_pairs))

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sd = self.supply_data
        if self.random_start:
            latest = max(self.start_week, self.end_week - self.min_episode_weeks)
            self.current_week = int(self.np_random.integers(self.start_week, latest + 1))
        else:
            self.current_week = self.start_week
        self.pair_index = 0
        self._new_order()
        self.balance = self.initial_balance
        self.inventory = sd.stock[:, :, self.start_week].astype(np.float64).copy()
        
        self.in_transit = np.zeros((sd.n_stores, sd.n_drugs))
        # arrivals[s, d, w] = units due to arrive in week w
        self.arrivals = np.zeros((sd.n_stores, sd.n_drugs, sd.n_weeks + self.max_lead + 2))
        return self.get_observation(), {}

    def inventory_transition(self, store_idx, drug_idx, action):
        sd, c = self.supply_data, self.current_week
        next_week = c + 1
        expedited = is_expedited(action)

        # 1. order quantity from the action
        forecast = self.get_forecast(store_idx, drug_idx)
        order_quantity = int(np.ceil(get_order_quantity(action, forecast)))

        # 2. cost per unit (expedited costs more)
        unit_cost = float(sd.cost[store_idx, drug_idx, c])
        cost_per_unit = unit_cost * (1 + EXPEDITE_COST) if expedited else unit_cost

        # 3. the shared pool decides what we can afford
        if order_quantity > 0:
            order_quantity = min(order_quantity, int(self.balance // cost_per_unit))
        order_cost = order_quantity * cost_per_unit
        self.balance -= order_cost

        # 4. schedule delivery
        delivery_time = calculate_delivery_time(int(sd.lead_time[store_idx]), expedited)
        if order_quantity > 0:
            self.arrivals[store_idx, drug_idx, c + delivery_time] += order_quantity
            self.in_transit[store_idx, drug_idx] += order_quantity

        # 5. receive what is due next week
        arrived = self.arrivals[store_idx, drug_idx, next_week]
        self.arrivals[store_idx, drug_idx, next_week] = 0.0
        self.in_transit[store_idx, drug_idx] -= arrived
        available = self.inventory[store_idx, drug_idx] + arrived

        # 6. serve next week's demand, and sales pay back into the pool
        demand = float(sd.demand[store_idx, drug_idx, next_week])
        dispensed = min(available, demand)
        unmet = max(demand - dispensed, 0.0)
        self.inventory[store_idx, drug_idx] = available - dispensed
        self.balance += dispensed * float(sd.price[store_idx, drug_idx, next_week])

        return dict(demand=demand, units_dispensed=dispensed, unmet_demand=unmet,
                    stockout=int(unmet > 0), order_quantity=order_quantity,
                    order_cost=order_cost, arrived_quantity=arrived,
                    delivery_time=delivery_time)

    def step(self, action):
        action = int(action)
        store_idx, drug_idx = self.pairs[self.pair_index]

        # expedite only when the rules allow it, otherwise use the normal-speed version
        expedite_denied = False
        if is_expedited(action) and not allow_expedite(self.state):
            action -= 3
            expedite_denied = True

        # weekly budget of this pair = what one normal week of its demand costs to buy.
        # Using this (not the pool size) keeps the cost penalty meaningful whatever the starting capital.
        weekly_budget = float(self.supply_data.baseline[drug_idx]
                              * self.supply_data.cost[store_idx, drug_idx, self.current_week])

        previous_coverage = self.get_inventory_coverage(store_idx, drug_idx)
        result = self.inventory_transition(store_idx, drug_idx, action)
        current_coverage = self.get_inventory_coverage(store_idx, drug_idx)

        reward = calculate_reward(
            demand=result["demand"],
            units_dispensed=result["units_dispensed"],
            unmet_demand=result["unmet_demand"],
            order_cost=result["order_cost"],
            weekly_budget=weekly_budget,
            previous_stockout_risk=risk_from_coverage(previous_coverage),
            current_stockout_risk=risk_from_coverage(current_coverage),
            previous_inventory_coverage=previous_coverage,
            current_inventory_coverage=current_coverage,
            expedited=is_expedited(action))

        # move to the next pair, and to the next week after the last pair
        self.pair_index += 1
        if self.pair_index >= self.n_pairs:
            self.pair_index = 0
            self.current_week += 1
        terminated = self.current_week >= self.end_week
        truncated = False

        info = dict(store_idx=store_idx, drug_idx=drug_idx, action=action,
                    expedite_denied=expedite_denied, capital_remaining=self.balance,
                    reward=reward, **result)
        return self.get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"week {self.current_week:3d} | pair {self.pair_index:3d}/{self.n_pairs} "
                  f"| capital {self.balance:14,.0f}")


# register the environment with an ID; the data is passed later, in gymnasium.make(...)
if "DigitalHealth-v0" not in gymnasium.registry:
    register(id="DigitalHealth-v0", entry_point=DigitalHealthEnv)


# ----------------------------------------------------------------------
# 5. Q-LEARNING AGENT
# ----------------------------------------------------------------------
LEARNING_RATE = 0.02
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPISODES = 50           # one episode = 116 weeks x 304 pairs = about 35,000 steps

Q_TABLE = np.zeros((int(np.prod(STATE_SIZE)), N_ACTIONS), dtype=np.float32)


def state_index(state):
    return int(np.ravel_multi_index(tuple(int(x) for x in state), STATE_SIZE))


def update_q_table(state, action, reward, next_state, terminated):
    s, ns = state_index(state), state_index(next_state)
    current_q = Q_TABLE[s, action]
    if terminated:
        target = reward
    else:
        target = reward + DISCOUNT_FACTOR * np.max(Q_TABLE[ns, get_valid_actions(next_state)])
    Q_TABLE[s, action] = current_q + LEARNING_RATE * (target - current_q)


def choose_action(state, epsilon):
    valid = get_valid_actions(state)
    if np.random.random() < epsilon:
        return int(np.random.choice(valid))
    q = Q_TABLE[state_index(state), valid]
    best = [a for a, v in zip(valid, q) if v == q.max()]
    return int(np.random.choice(best))                  # ties broken at random


def train_agent(env, episodes=EPISODES):
    Q_TABLE[:] = 0.0
    decay = (EPSILON_MIN / EPSILON_START) ** (1.0 / max(0.8 * episodes, 1))
    epsilon, episode_rewards = EPSILON_START, []

    for episode in range(episodes):
        obs, _ = env.reset()
        state, total_reward, terminated = tuple(obs), 0.0, False
        while not terminated:
            action = choose_action(state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(obs)
            update_q_table(state, action, reward, next_state, terminated or truncated)
            total_reward += reward
            state = next_state
        epsilon = max(EPSILON_MIN, epsilon * decay)
        episode_rewards.append(total_reward)
        print(f"episode {episode + 1:3d} | reward {total_reward:11.1f} "
              f"| epsilon {epsilon:.3f} | capital {info['capital_remaining']:,.0f}")
    return episode_rewards


def greedy_policy(state):
    valid = get_valid_actions(state)
    return int(valid[int(np.argmax(Q_TABLE[state_index(state), valid]))])


def evaluate(env, policy):
    obs, _ = env.reset()
    state, terminated = tuple(obs), False
    total = dict(reward=0.0, demand=0.0, dispensed=0.0, stockouts=0, steps=0,
                 order_spend=0.0, expedited=0)
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(policy(state))
        state = tuple(obs)
        total["reward"] += reward
        total["demand"] += info["demand"]
        total["dispensed"] += info["units_dispensed"]
        total["stockouts"] += info["stockout"]
        total["order_spend"] += info["order_cost"]
        total["expedited"] += int(is_expedited(info["action"]))
        total["steps"] += 1
    # cash alone can mislead: money spent on stock is not lost, so also report what the stock is worth
    raw = env.unwrapped
    week = min(raw.current_week, raw.supply_data.n_weeks - 1)
    stock_value = float(((raw.inventory + raw.in_transit) * raw.supply_data.cost[:, :, week]).sum())
    return dict(reward=round(total["reward"], 1),
                service_level=round(float(total["dispensed"] / max(total["demand"], 1)), 3),
                stockout_rate=round(total["stockouts"] / total["steps"], 3),
                expedited_orders=total["expedited"],
                final_capital=round(info["capital_remaining"]),
                stock_value=round(stock_value),
                net_worth=round(info["capital_remaining"] + stock_value))


if __name__ == "__main__":
    supply = SupplyData()
    supply.summary()

    cut = int(supply.n_weeks * 0.75)                    # weeks 0-116 train, 117-155 test
    env_train = gymnasium.make("DigitalHealth-v0", supply_data=supply,
                               start_week=0, end_week=cut - 1,shuffle_pairs=True, random_start=True)
    env_test = gymnasium.make("DigitalHealth-v0", supply_data=supply,
                              start_week=cut, end_week=supply.n_weeks - 1)
    print("shared starting capital:", f"{env_train.unwrapped.initial_balance:,.0f}")

    train_agent(env_train)

    print("\nresults on unseen weeks")
    print("Q-learning    :", evaluate(env_test, greedy_policy))
    print("always hold   :", evaluate(env_test, lambda s: 0))
    print("always medium :", evaluate(env_test, lambda s: 2))
    print("always large  :", evaluate(env_test, lambda s: 3))
