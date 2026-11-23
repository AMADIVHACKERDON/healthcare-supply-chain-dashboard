import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class DigitalHealthInvestmentEnv(gym.Env):
    """
    Custom Environment for Digital Health Capital Allocation
    States: [EHR_demand, Supply_chain_efficiency, Wearable_engagement, Surveillance_risk, Available_capital]
    Actions: [Allocate_to_Clinical_AI, Allocate_to_Logistics, Allocate_to_Remote_Care]
    """
    metadata = {"render_modes": ["human"]}

    def init(self):
        super(DigitalHealthInvestmentEnv, self).init()
        
        # Continuous Action Space: Allocation percentages for 3 target sectors (must sum <= 1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation Space: 5 key metrics normalized between 0 and 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Simulate initial states from your 4 data structures + capital
        self.state = np.array([
            0.65,  # EHR data: High chronic disease prevalence
            0.50,  # Supply Chain: Moderate logistical bottlenecks
            0.75,  # Wearables: High user/sensor engagement trends
            0.20,  # Surveillance: Low immediate pandemic risk
            1.00   # Capital Vector: 100% of initial fund available
        ], dtype=np.float32)
        
        return self.state, {}

    def step(self, action):
        # Normalize actions so they don't exceed remaining available capital
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum * min(action_sum, self.state[4])
        
        ehr_demand, supply_eff, wearable_eng, surv_risk, capital = self.state
        alloc_clinical, alloc_logistics, alloc_remote = action

        # Simulate Environment Transitions (How investments alter the data structures)
        new_ehr_demand = max(0.0, ehr_demand - (alloc_clinical * 0.2)) # Investment reduces unmet demand
        new_supply_eff = min(1.0, supply_eff + (alloc_logistics * 0.3)) # Logistics funding fixes bottlenecks
        new_wearable_eng = min(1.0, wearable_eng + (alloc_remote * 0.15))
        new_surv_risk = max(0.0, surv_risk + np.random.normal(0, 0.05)) # Exogenous variance
        new_capital = max(0.0, capital - np.sum(action))

        # Reward Function: Multi-Objective optimization
        # Rewards heavy deployment when demand/risk is high, penalizes wasting capital with low engagement
        clinical_roi = alloc_clinical * ehr_demand
        logistics_roi = alloc_logistics * (1.0 - supply_eff) # Higher reward for fixing bad supply chains
        remote_roi = alloc_remote * wearable_eng * (1.0 + surv_risk)
        
        reward = float(clinical_roi + logistics_roi + remote_roi - (new_capital * 0.01))

        # Update state
        self.state = np.array([new_ehr_demand, new_supply_eff, new_wearable_eng, new_surv_risk, new_capital], dtype=np.float32)
        
        # Terminate if all capital spent or steps exhausted
        terminated = bool(self.state[4] <= 0.01)
        truncated = False # Add step-count logic if managing time-horizons
        
        return self.state, reward, terminated, truncated, {}

# --- Execution & Agent Training ---
if name == "main":
    # Initialize the custom data-driven environment
    env = DigitalHealthInvestmentEnv()
    
    # Use Proximal Policy Optimization (PPO) for continuous action-space allocation
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Training RL Agent on Digital Health Portfolios...")
    model.learn(total_timesteps=20000)
    
    # Test the trained agent's capital deployment strategy
    obs, _ = env.reset()
    print(f"\nInitial Data State: {obs}")
    action, _states = model.predict(obs, deterministic=True)
    print(f"Optimal Capital Allocation Selected by Agent: {action}")