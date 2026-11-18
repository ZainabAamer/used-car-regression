# Used car prices: should the dealer buy this car?

Linear regression (statsmodels OLS) predicting used car sale price, then testing a buy/pass rule on 40 auction lots.

## Files
- `analysis.py`: the whole analysis. Reproduces every number on the slides.
- `used_car_sales.csv`: 240 cars sold in the last year (model data).
- `auction_lots.csv`: 40 auction lots used only to test the decision rule.
- `requirements.txt`: packages needed.
- Outputs after running: `pred_vs_actual.png`, `auction_decisions.csv`, `results.json`.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Method
1. Random 70/30 split of the 240 cars (seed 42): 168 train, 72 test.
2. v1: `Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident_Yes + Num_Owners` fit on train. `Accident_Yes` is 1 if the history report shows an accident (baseline: no accident).
3. v2: drops any feature with p >= 0.05. The version handed over is chosen by training adjusted R-squared.
4. Fit evaluated on the 72 test cars: R-squared and RMSE, baseline (training mean) vs model.
5. Decision rule: buy if auction price + $1,500 < predicted sale price. Applied to the 40 lots; profit = Later_Sale_Price - Auction_Price - 1,500.

Note: the model called v2 in analysis.py (Num_Owners dropped) is the final model. It is called v1 on the slides, per the assignment.
