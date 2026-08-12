import pandas as pd
from config import DATA_DIR


def load_data():
    customers = pd.read_csv(
        f"{DATA_DIR}/customers.csv"
    )

    usage = pd.read_csv(
        f"{DATA_DIR}/customer_usage.csv"
    )

    billing = pd.read_csv(
        f"{DATA_DIR}/billing_records.csv"
    )

    csat = pd.read_csv(
        f"{DATA_DIR}/csat_feedback.csv"
    )

    support = pd.read_csv(
        f"{DATA_DIR}/support_interactions.csv"
    )

    return {
        "customers": customers,
        "usage": usage,
        "billing": billing,
        "csat": csat,
        "support": support,
    }


if __name__ == "__main__":
    data = load_data()

    for name, df in data.items():
        print(f"{name}: {df.shape}")