import numpy as np
def strategy_to_action(strategy, observation):

        n_drugs = 19
        n_stores = 16

    # Extract the current observation features.
        forecast = observation["forecast_demand"]
        risk = observation["stockout_risk"]
        disease = observation["disease_demand"]
        coverage = observation["inventory_coverage"]

        # Convert all features to NumPy arrays.
        forecast = np.asarray(forecast, dtype=np.float32)
        risk = np.asarray(risk, dtype=np.float32)
        disease = np.asarray(disease, dtype=np.float32)
        coverage = np.asarray(coverage, dtype=np.float32)

        # Normalize features to the range 0–1.
        def normalize(values):
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            maximum = values.max()

            if maximum > 0:
                return values / maximum

            return np.zeros_like(values)

        forecast_score = normalize(forecast)
        risk_score = normalize(risk)
        disease_score = normalize(disease)

        # Lower coverage means greater need for inventory.
        coverage_score = 1.0 - normalize(coverage)

        # Convert store-drug features into drug-level priorities.
        drug_risk = risk_score.max(axis=0)
        drug_disease = disease_score.max(axis=0)

        # Initialize the two continuous action components.
        order_actions = np.zeros(n_drugs, dtype=np.float32)

        allocation_actions = np.zeros(
            (n_drugs, n_stores),
            dtype=np.float32
        )

        # Strategy 0: No procurement.
        if strategy == 0:
            pass

        # Strategy 1: Conservative procurement.
        elif strategy == 1:
            order_actions[:] = 0.25
            allocation_actions = forecast_score.T

        # Strategy 2: Balanced procurement.
        elif strategy == 2:
            order_actions[:] = 0.50

            allocation_actions = (
                forecast_score * 0.5
                + risk_score * 0.5
            ).T

        # Strategy 3: Aggressive procurement.
        elif strategy == 3:
            order_actions[:] = 0.90

            allocation_actions = (
                forecast_score * 0.5
                + risk_score * 0.5
            ).T

        # Strategy 4: Stockout-risk prioritisation.
        elif strategy == 4:
            order_actions = 0.30 + (0.70 * drug_risk)

            allocation_actions = (
                risk_score * 0.7
                + forecast_score * 0.3
            ).T

        # Strategy 5: Disease-demand prioritisation.
        elif strategy == 5:
            order_actions = 0.30 + (0.70 * drug_disease)

            allocation_actions = (
                disease_score * 0.7
                + forecast_score * 0.3
            ).T

        # Strategy 6: Emergency procurement.
        elif strategy == 6:
            order_actions = 0.70 + (0.30 * drug_risk)

            allocation_actions = (
                risk_score * 0.5
                + disease_score * 0.3
                + coverage_score * 0.2
            ).T

        # Strategy 7: Capital preservation.
        elif strategy == 7:
            order_actions = 0.15 + (0.20 * drug_risk)

            allocation_actions = (
                risk_score * 0.6
                + coverage_score * 0.4
            ).T

        else:
            raise ValueError(
                f"Unknown DQN strategy: {strategy}"
            )

        # Ensure every action is within the environment's Box limits.
        order_actions = np.clip(
            order_actions, 0.0, 1.0
        )

        allocation_actions = np.clip(
            allocation_actions, 0.0, 1.0
        )

        # Flatten allocation and combine both action components.
        continuous_action = np.concatenate([
            order_actions,
            allocation_actions.flatten()
        ]).astype(np.float32)

        return continuous_action
