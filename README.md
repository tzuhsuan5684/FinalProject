# Ubike 借閱量預測專案 (Ubike Demand Prediction)

本專案使用多種機器學習模型（Linear Regression、Random Forest、XGBoost）預測台北市 Ubike 各站點的借閱量。結合時間特徵、氣候數據與站點周邊靜態地理資訊，並提供批次實驗、殘差分析與模型儲存功能，方便比較與部署。

## 📂 專案結構

```text
.
├── FINAL_MODEL_DATA_WITH_FEATURES.csv   # 主要資料 (時間、氣候、地理、滯後特徵)
├── main.py                               # 單次訓練與比較，並儲存模型到 model/
├── run_experiments.py                    # 批次測試多種特徵組合與模型
├── rf_residual_analysis.py               # 使用指定特徵進行 RF 殘差分析與時間平均圖
├── batch_experiment_results.csv          # 批次實驗結果 (MAE, RMSE, R2)
├── results/                              # 產出圖表與指標
│   └── rf_residual_analysis/
│       ├── residuals_by_hour.png
│       ├── residuals_vs_predicted.png
│       ├── residual_distribution.png
│       └── time_average_comparison.png
└── model/                                # 儲存訓練完成的模型 (.joblib)
    ├── linear_regression_model.joblib
    ├── random_forest_model.joblib
    └── xgboost_model.joblib
```

## 🚀 如何執行

### 1. 執行單次訓練並儲存模型 (main)
使用完整資料的預處理（數值補中位數+標準化、類別補眾數+OneHot），訓練 Linear/RandomForest/XGBoost 並儲存模型：

```powershell
python main.py
```

產出：
- `results/baseline_model_metrics.csv` 與 `baseline_model_comparison.png`
- `model/linear_regression_model.joblib`
- `model/random_forest_model.joblib`
- `model/xgboost_model.joblib`

### 2. 執行批次實驗 (run_experiments)
測試多組特徵與模型，並可自動載入 `main.py` 儲存的模型設定：

```powershell
python run_experiments.py
```

說明：
- 以 3 月資料為測試集，其餘月份為訓練集。
- 每組特徵都會訓練並評估三種模型，結果寫入 `batch_experiment_results.csv`。
- 若 `model/xxx_model.joblib` 存在，會載入並使用該模型的設定（確保一致性）。

### 3. RF 殘差分析與時間平均折線圖 (rf_residual_analysis)
使用固定特徵：`['hour','weekday','is_weekend','is_peak','rent_count_lag_3','rent_count_lag_24']` 進行 RF 預測之殘差分析，並繪製每小時的實際 vs 預測平均折線圖。

```powershell
python rf_residual_analysis.py
```

說明：
- 若 `model/rf_model.joblib` 存在則載入；否則會重新訓練並儲存。
- `residuals_by_hour.png` 使用紅藍漸層（`RdBu`）。

## 📊 實驗設計重點

1.  **資料切割 (Data Splitting)**：
    *   採用 **Time-based Split** 而非 Random Split。
    *   **測試集**：3 月份完整資料。
    *   **訓練集**：其餘月份資料。
    *   *目的：模擬真實預測情境，避免 Data Leakage。*

2.  **特徵工程 (Feature Engineering)**：
    *   **動態特徵**：`hour`, `weekday`, `temperature`, `rainfall`, `wind_speed`, `is_weekend`, `is_peak`
    *   **滯後特徵**：`rent_count_lag_3`, `rent_count_lag_24`
    *   **靜態特徵**：`mrt_dist_nearest_m`, `school_dist_nearest_m`, `park_dist_nearest_m`, `population_count`, `Quantity`, `latitude`, `longitude`


