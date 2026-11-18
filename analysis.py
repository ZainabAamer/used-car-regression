"""
Used car regression: should the dealer buy this car?

Run:  python analysis.py
Needs used_car_sales.csv and auction_lots.csv in the same folder as this file.
Every number on the slides comes from what this script prints / saves.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
SEED = 42            # fixed so the random split is the same every run
TEST_FRAC = 0.30     # 30% of the 240 cars held out
RECON_COST = 1500    # cost to get a car lot-ready
ALPHA = 0.05

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# ---------------------------------------------------------------- load
for name in ("used_car_sales.csv", "auction_lots.csv"):
    if not (HERE / name).exists():
        sys.exit(f"Missing {name}. Put it in the same folder as analysis.py.")

cars = pd.read_csv(HERE / "used_car_sales.csv")
lots = pd.read_csv(HERE / "auction_lots.csv")


def prep(df):
    """Turn the yes/no Accident column into a 0/1 dummy (no = baseline)."""
    df = df.copy()
    df["Accident_Yes"] = (df["Accident"].str.strip().str.lower() == "yes").astype(int)
    return df


cars, lots = prep(cars), prep(lots)

print("=" * 70)
print("DATA CHECK")
print("used_car_sales shape:", cars.shape, "| auction_lots shape:", lots.shape)
print("missing values in cars:", int(cars.isna().sum().sum()))
print(cars.describe().T)
print("Accident counts:\n", cars["Accident"].value_counts())

# ---------------------------------------------------------------- 1. split
test = cars.sample(frac=TEST_FRAC, random_state=SEED)
train = cars.drop(test.index)
print("=" * 70)
print(f"SPLIT: train = {len(train)} cars, test = {len(test)} cars (seed {SEED})")

# ---------------------------------------------------------------- 2. fit
V1_FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes + Num_Owners"
m1 = smf.ols(V1_FORMULA, data=train).fit()
print("=" * 70)
print("MODEL v1 (all five features), fit on TRAIN only")
print(m1.summary())
print("\nCoefficients and p-values:")
print(pd.DataFrame({"coef": m1.params, "p_value": m1.pvalues}))
print(f"\nv1 R2 = {m1.rsquared:.4f} | adjusted R2 = {m1.rsquared_adj:.4f}")

# second version: drop any feature that is not significant (only if there is one)
features = [t for t in m1.params.index if t != "Intercept"]
weak = [t for t in features if m1.pvalues[t] >= ALPHA]
m2, V2_FORMULA = None, None
if weak:
    keep = [t for t in features if t not in weak]
    V2_FORMULA = "Sale_Price ~ " + " + ".join(keep)
    m2 = smf.ols(V2_FORMULA, data=train).fit()
    print("=" * 70)
    print(f"MODEL v2: dropped {weak} (p >= {ALPHA})")
    print(pd.DataFrame({"coef": m2.params, "p_value": m2.pvalues}))
    print(f"v2 R2 = {m2.rsquared:.4f} | adjusted R2 = {m2.rsquared_adj:.4f}")
else:
    print("\nAll features significant at 0.05, so no v2 is needed.")

# pick the version to hand over: higher adjusted R2 on the training fit (ties -> simpler)
if m2 is not None and m2.rsquared_adj >= m1.rsquared_adj - 0.001:
    final, final_formula, final_label = m2, V2_FORMULA, "v2 (dropped: " + ", ".join(weak) + ")"
else:
    final, final_formula, final_label = m1, V1_FORMULA, "v1 (all five features)"
print(f"\nVERSION HANDED OVER: {final_label}")
print("Formula:", final_formula)

# ---------------------------------------------------------------- 3. test-set fit
baseline_value = train["Sale_Price"].mean()
y_test = test["Sale_Price"]
pred_base = pd.Series(baseline_value, index=test.index)
pred_model = final.predict(test)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


r2_base, r2_model = r2(y_test, pred_base), r2(y_test, pred_model)
rmse_base, rmse_model = rmse(y_test, pred_base), rmse(y_test, pred_model)
print("=" * 70)
print(f"TEST-SET FIT ({len(test)} cars the model never saw)")
print(f"baseline (train mean ${baseline_value:,.0f}): R2 = {r2_base:.3f} | RMSE = ${rmse_base:,.0f}")
print(f"model {final_label}: R2 = {r2_model:.3f} | RMSE = ${rmse_model:,.0f}")

# chart: predicted vs actual on test cars
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.scatter(y_test, pred_model, color="black", s=22, label="Model prediction")
ax.scatter(y_test, pred_base, color="#999999", s=22, marker="x", label="Baseline (same price every car)")
lo = min(y_test.min(), pred_model.min(), pred_base.min())
hi = max(y_test.max(), pred_model.max(), pred_base.max())
ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="Perfect prediction")
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.set_title(f"Test cars: predicted vs actual (n={len(test)})")
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "pred_vs_actual.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- 4. decision rule
def predict_price(mileage_k, age_years, horsepower, accident, num_owners):
    """Predicted retail sale price for one car. accident is 'yes' or 'no'."""
    row = pd.DataFrame({
        "Mileage_K": [mileage_k], "Age_Years": [age_years], "Horsepower": [horsepower],
        "Accident_Yes": [1 if str(accident).strip().lower() == "yes" else 0],
        "Num_Owners": [num_owners],
    })
    return float(final.predict(row).iloc[0])


def buy_or_pass(mileage_k, age_years, horsepower, accident, num_owners, auction_price):
    """BUY if auction price + $1,500 is below the predicted sale price, else PASS."""
    predicted = predict_price(mileage_k, age_years, horsepower, accident, num_owners)
    return "BUY" if auction_price + RECON_COST < predicted else "PASS"


lots["Predicted_Price"] = final.predict(lots)
lots["Model_Decision"] = [
    buy_or_pass(r.Mileage_K, r.Age_Years, r.Horsepower, r.Accident, r.Num_Owners, r.Auction_Price)
    for r in lots.itertuples()
]
lots["Baseline_Decision"] = np.where(lots["Auction_Price"] + RECON_COST < baseline_value, "BUY", "PASS")
lots["Profit_If_Bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - RECON_COST


def score(col):
    bought = lots[lots[col] == "BUY"]
    return {
        "cars_bought": int(len(bought)),
        "money_losing": int((bought["Profit_If_Bought"] < 0).sum()),
        "total_profit": float(bought["Profit_If_Bought"].sum()),
    }


res_base, res_model = score("Baseline_Decision"), score("Model_Decision")
print("=" * 70)
print(f"DECISION RULE on the {len(lots)} auction lots (profit = Later_Sale - Auction - ${RECON_COST:,})")
print(f"baseline rule: {res_base}")
print(f"model rule   : {res_model}")
print(f"Model minus baseline profit: ${res_model['total_profit'] - res_base['total_profit']:,.0f}")
print("(For reference, profit if the dealer bought ALL lots: "
      f"${lots['Profit_If_Bought'].sum():,.0f})")

lots.to_csv(HERE / "auction_decisions.csv", index=False)

# ---------------------------------------------------------------- save numbers
out = {
    "n_cars": int(len(cars)), "n_train": int(len(train)), "n_test": int(len(test)), "seed": SEED,
    "formula_v1": V1_FORMULA, "formula_v2": V2_FORMULA, "handed_over": final_label,
    "v1_adj_r2_train": float(m1.rsquared_adj),
    "v2_adj_r2_train": None if m2 is None else float(m2.rsquared_adj),
    "final_adj_r2_train": float(final.rsquared_adj),
    "coefficients": {k: float(v) for k, v in final.params.items()},
    "p_values": {k: float(v) for k, v in final.pvalues.items()},
    "baseline_value": float(baseline_value),
    "test": {"r2_baseline": float(r2_base), "r2_model": float(r2_model),
             "rmse_baseline": rmse_base, "rmse_model": rmse_model},
    "auction_baseline": res_base, "auction_model": res_model,
}
(HERE / "results.json").write_text(json.dumps(out, indent=2))
print("\nSaved: pred_vs_actual.png, auction_decisions.csv, results.json")
