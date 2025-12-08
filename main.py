import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# ==========================================
# 1. 設定與資料載入
# ==========================================
RANDOM_STATE = 42
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("📚 正在載入資料...")
df = pd.read_csv('FINAL_MODEL_DATA_WITH_FEATURES.csv', parse_dates=['rent_time'])

# ==========================================
# 2. 資料前處理與切分
# ==========================================
print("✂️  正在切分訓練集與測試集...")

# 準備特徵與目標
target = df['rent_count']
features = df.drop(columns=['rent_count', 'rent_time'], errors='ignore')

# 依時間切分 (3月為測試集)
test_mask = df['rent_time'].dt.month == 3
X_train = features[~test_mask]
y_train = target[~test_mask]
X_test = features[test_mask]
y_test = target[test_mask]

print(f"   訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")

# 建立預處理器 (數值補中位數+標準化, 類別補眾數+OneHot)
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# ==========================================
# 3. 模型定義
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=RANDOM_STATE
    )
}

# ==========================================
# 4. 訓練與評估
# ==========================================
results = []

print("\n🚀 開始訓練模型...")
for name, model in models.items():
    print(f"   正在訓練 {name}...")
    
    # 建立並訓練 Pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # 儲存模型
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()
    model_path = os.path.join(model_dir, f'{safe_name}_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"   💾 模型已儲存至: {model_path}")

    # 預測
    y_pred = pipeline.predict(X_test)
    
    # 計算指標
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

# 轉為 DataFrame 並顯示
metrics_df = pd.DataFrame(results).sort_values('MAE')
print("\n📊 模型評估結果:")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 儲存結果
metrics_path = os.path.join(RESULTS_DIR, 'baseline_model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"\n💾 評估表已儲存至: {metrics_path}")

# ==========================================
# 5. 繪製比較圖表
# ==========================================
print("🎨 正在繪製比較圖表...")
sns.set_theme(style="whitegrid")

# 準備繪圖資料
long_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
error_df = long_df[long_df["Metric"].isin(["MAE", "RMSE"])]
r2_df = long_df[long_df["Metric"] == "R2"]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左圖: MAE & RMSE
sns.barplot(data=error_df, x="Metric", y="Score", hue="Model", ax=axes[0], palette="viridis")
axes[0].set_title("Error Metrics (Lower is Better)")
axes[0].set_ylabel("Score")

# 右圖: R2
sns.barplot(data=r2_df, x="Metric", y="Score", hue="Model", ax=axes[1], palette="viridis")
axes[1].set_title("R2 Score (Higher is Better)")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1.0)

# 標註數值
for ax in axes:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

plt.tight_layout()
figure_path = os.path.join(RESULTS_DIR, 'baseline_model_comparison.png')
plt.savefig(figure_path, dpi=300)
print(f"🖼️  比較圖已儲存至: {figure_path}")
print("\n🎉 執行完畢！")
