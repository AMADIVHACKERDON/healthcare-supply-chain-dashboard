import numpy as np
def strategy_to_action(self, strategy, observation):

    n_drugs = self.env.unwrapped.n_drugs
    n_stores = self.env.unwrapped.n_stores

    forecast = np.asarray(
        observation["forecast_demand"],
        dtype=np.float32
    )

    risk = np.asarray(
        observation["stockout_risk"],
        dtype=np.float32
    )

    disease = np.asarray(
        observation["disease_demand"],
        dtype=np.float32
    )

    coverage = np.asarray(
        observation["inventory_coverage"],
        dtype=np.float32
    )

    investment_profiles = {
        0: [0.25, 0.25, 0.25, 0.25],
        1: [0.40, 0.25, 0.20, 0.15],
        2: [0.20, 0.40, 0.20, 0.20],
        3: [0.20, 0.20, 0.40, 0.20],
        4: [0.20, 0.20, 0.20, 0.40],
        5: [0.35, 0.35, 0.15, 0.15],
        6: [0.15, 0.15, 0.35, 0.35],
        7: [0.20, 0.30, 0.30, 0.20]
    }

    if strategy not in investment_profiles:
        raise ValueError(f"Invalid strategy: {strategy}")

    investment_actions = np.asarray(
        investment_profiles[strategy],
        dtype=np.float32
    )

    # Aggregate store-level information by drug.
    drug_risk = np.mean(risk, axis=0)
    drug_disease = np.mean(disease, axis=0)
    drug_coverage = np.mean(coverage, axis=0)

    drug_forecast = np.sum(forecast, axis=0)

    order_actions = np.zeros(n_drugs, dtype=np.float32)

    allocation_actions = np.zeros(
        (n_drugs, n_stores),
        dtype=np.float32
    )

    for drug_idx in range(n_drugs):

        if strategy == 0:
            intensity = 0.0

        elif strategy == 1:
            intensity = 0.25

        elif strategy == 2:
            intensity = 0.50

        elif strategy == 3:
            intensity = 1.0

        elif strategy == 4:
            intensity = float(np.clip(
                drug_risk[drug_idx], 0.0, 1.0
            ))

        elif strategy == 5:
            intensity = float(np.clip(
                drug_disease[drug_idx], 0.0, 1.0
            ))

        elif strategy == 6:
            intensity = 1.0

        else:  # Strategy 7: capital preservation
            intensity = 0.25 if drug_coverage[drug_idx] < 1.0 else 0.0

        order_actions[drug_idx] = intensity

        for store_idx in range(n_stores):

            if strategy == 4:
                priority = risk[store_idx, drug_idx]

            elif strategy == 5:
                priority = disease[store_idx, drug_idx]

            elif strategy == 6:
                priority = (
                    risk[store_idx, drug_idx]
                    + disease[store_idx, drug_idx]
                )

            else:
                priority = 1.0

            allocation_actions[drug_idx, store_idx] = (
                intensity
                * forecast[store_idx, drug_idx]
                * max(float(priority), 0.0)
            )

    continuous_action = np.concatenate([
        investment_actions,
        order_actions,
        allocation_actions.flatten()
    ]).astype(np.float32)

    return continuous_action
