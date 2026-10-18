"""Generate synthetic healthcare member data with churn labels."""

import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

NUM_MEMBERS = 5000
CHURN_RATE = 0.15
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PLAN_TYPES = ["HMO", "PPO", "EPO", "HDHP"]


def generate():
    rows = []

    for i in range(1, NUM_MEMBERS + 1):
        age = random.randint(18, 80)
        region = random.choice(REGIONS)
        plan_type = random.choices(PLAN_TYPES, weights=[0.35, 0.30, 0.15, 0.20])[0]
        tenure_months = random.randint(1, 120)
        gender = random.choice(["M", "F"])
        income_bracket = random.choices(
            ["low", "medium", "high"],
            weights=[0.30, 0.45, 0.25],
        )[0]

        # utilization
        annual_claims = max(0, int(np.random.poisson(3 + age / 25)))
        er_visits = max(0, int(np.random.poisson(0.3 if age > 50 else 0.15)))
        rx_fills = max(0, int(np.random.poisson(2 + age / 30)))
        preventive_visits = max(0, int(np.random.poisson(1.0)))
        specialist_visits = max(0, int(np.random.poisson(1.5 if age > 40 else 0.5)))

        # engagement
        portal_logins = max(0, int(np.random.poisson(5 if age < 45 else 2)))
        mobile_sessions = max(0, int(np.random.poisson(8 if age < 35 else 2)))
        calls_to_service = max(0, int(np.random.poisson(2.5)))
        grievances_filed = max(0, int(np.random.poisson(0.3)))

        # cost
        total_cost = max(100, np.random.lognormal(mean=7.5 + annual_claims * 0.1, sigma=0.8))
        member_oop = total_cost * np.random.uniform(0.1, 0.35)
        premium = np.random.uniform(200, 800) if plan_type != "HDHP" else np.random.uniform(100, 400)

        # satisfaction
        satisfaction = np.clip(
            np.random.normal(7.0 - calls_to_service * 0.2 - grievances_filed * 1.0, 1.5),
            1, 10,
        )
        nps_response = int(np.clip(np.random.normal(satisfaction - 1, 2), 0, 10)) if random.random() < 0.4 else None

        # churn probability — driven by satisfaction, grievances, tenure, cost burden
        churn_logit = (
            -2.0
            + (-0.3 * satisfaction)
            + (0.8 * grievances_filed)
            + (-0.01 * tenure_months)
            + (0.3 * calls_to_service)
            + (-0.05 * portal_logins)
            + (0.0001 * member_oop)
            + (0.2 if income_bracket == "low" else 0)
        )
        churn_prob = 1 / (1 + np.exp(-churn_logit))
        churned = int(random.random() < churn_prob)

        rows.append({
            "member_id": f"MBR-{i:05d}",
            "age": age,
            "gender": gender,
            "region": region,
            "plan_type": plan_type,
            "income_bracket": income_bracket,
            "tenure_months": tenure_months,
            "annual_claims": annual_claims,
            "er_visits": er_visits,
            "rx_fills": rx_fills,
            "preventive_visits": preventive_visits,
            "specialist_visits": specialist_visits,
            "portal_logins_6m": portal_logins,
            "mobile_sessions_6m": mobile_sessions,
            "calls_to_service_6m": calls_to_service,
            "grievances_filed": grievances_filed,
            "total_cost_ytd": round(total_cost, 2),
            "member_oop_ytd": round(member_oop, 2),
            "monthly_premium": round(premium, 2),
            "satisfaction_score": round(satisfaction, 1),
            "nps_score": nps_response,
            "churned": churned,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/member_churn_data.csv", index=False)

    churn_rate = df["churned"].mean()
    print(f"Members: {len(df):,}")
    print(f"Churn rate: {churn_rate:.1%}")
    print(f"Churned: {df['churned'].sum()}")
    print(f"Retained: {(~df['churned'].astype(bool)).sum()}")
    print(f"Features: {len(df.columns)}")
    print("Saved to data/member_churn_data.csv")
