import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import platform
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. 設定與準備
# ==========================================
DATA_FILE = 'FINAL_MODEL_DATA_CLEAN.csv'
RESULT_CSV = 'batch_experiment_results.csv'
OUTPUT_DIR = 'report_images'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 設定中文字型
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif system_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 資料載入與模型訓練 (使用最佳配置)
# ==========================================
print("🚀 正在載入資料並訓練最佳模型...")

df = pd.read_csv(DATA_FILE)
df['rent_time'] = pd.to_datetime(df['rent_time'])

# 定義最佳特徵組合
features = ['hour', 'weekday', 'Quantity', 'mrt_dist_nearest_m', 'school_dist_nearest_m', 'park_dist_nearest_m', 'population_count']
# 準備資料
X=df[features]
# X = pd.get_dummies(X, columns=['rent_station'], drop_first=True)
y = df['rent_count']

# 切分測試集 (3月)
is_test = df['rent_time'].dt.month == 3
X_train = X[~is_test]
X_test = X[is_test]
y_train = y[~is_test]
y_test = y[is_test]
test_time = df.loc[is_test, 'rent_time']

# 訓練 最佳模型 (RandomForestRegressor, depth=10)
model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

r2 = r2_score(y_test, y_pred)
print(f"✅ 模型訓練完成 (R2: {r2:.4f})")

# ==========================================
# 3. 繪圖函數
# ==========================================

def plot_feature_importance():
    """圖 1: 特徵重要性排行"""
    print("📊 繪製特徵重要性...")
    plt.figure(figsize=(12, 6))
    
    # 取得重要性並排序
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features[i] for i in indices]
    
    # 繪圖
    sns.barplot(x=importances[indices], y=names, palette='viridis')
    plt.title(f'影響借車量的關鍵因素 (Feature Importance)\nModel R2: {r2:.3f}')
    plt.xlabel('重要性分數')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_Feature_Importance.png', dpi=300)
    plt.close()

def plot_hourly_trend():
    """圖 2: 平均小時趨勢"""
    print("📊 繪製小時趨勢圖...")
    df_plot = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Hour': test_time.dt.hour
    })
    hourly_avg = df_plot.groupby('Hour').mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg.index, hourly_avg['Actual'], 'o-', label='真實數據 (Actual)', color='black', linewidth=2)
    plt.plot(hourly_avg.index, hourly_avg['Predicted'], 'o--', label='模型預測 (Predicted)', color='#d62728', linewidth=2)
    
    plt.title('平均每小時借車量趨勢 (24小時循環)')
    plt.xlabel('小時 (0-23)')
    plt.ylabel('平均借車數')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_Hourly_Trend.png', dpi=300)
    plt.close()

def plot_one_week_zoom():
    """圖 3: 一週預測細節"""
    print("📊 繪製一週細節圖...")
    # 取測試集的前 7 天 (假設資料是每小時一筆，7天約 168 筆，但因為有多個站點，我們取前 500 筆來示意，或者過濾出單一站點)
    # 為了圖表清晰，我們只畫出「全體平均」的時間序列，或者取前 200 個時間點
    
    # 這裡我們畫出「全體平均」隨時間的變化，這樣比較不亂
    df_plot = pd.DataFrame({
        'Time': test_time.values,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    # 依時間聚合平均 (因為同一時間有多個站點)
    time_agg = df_plot.groupby('Time').mean().reset_index()
    
    # 取前 7 天 (約 168 小時)
    subset = time_agg.head(168)
    
    plt.figure(figsize=(14, 6))
    plt.plot(subset['Time'], subset['Actual'], label='真實數據', color='black', alpha=0.6)
    plt.plot(subset['Time'], subset['Predicted'], label='預測數據', color='#1f77b4', linewidth=2)
    
    plt.title('一週內的借車量變化預測 (全站點平均)')
    plt.xlabel('時間')
    plt.ylabel('平均借車數')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_One_Week_Zoom.png', dpi=300)
    plt.close()

def plot_scatter():
    """圖 4: 散佈圖"""
    print("📊 繪製散佈圖...")
    plt.figure(figsize=(8, 8))
    
    # 為了避免點太多，隨機抽樣 1000 點
    indices = np.random.choice(len(y_test), size=min(1000, len(y_test)), replace=False)
    
    plt.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.3, color='#2ca02c')
    
    # 45度線
    p1 = max(y_test.max(), y_pred.max())
    p2 = min(y_test.min(), y_pred.min())
    plt.plot([p1, p2], [p1, p2], 'r--', linewidth=2, label='完美預測線')
    
    plt.title(f'真實值 vs 預測值 (R2={r2:.3f})')
    plt.xlabel('真實借車數')
    plt.ylabel('預測借車數')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_Scatter_Plot.png', dpi=300)
    plt.close()

def plot_experiment_comparison():
    """圖 5: 實驗比較圖 (讀取 batch_experiment_results.csv)"""
    if not os.path.exists(RESULT_CSV):
        print("⚠️ 找不到實驗結果 CSV，跳過比較圖。")
        return

    print("📊 繪製實驗比較圖...")
    try:
        res_df = pd.read_csv(RESULT_CSV)
        
        # 簡化特徵名稱以便繪圖
        res_df['Feature_Count'] = res_df['Features'].apply(lambda x: len(str(x).split(';')))
        res_df['Short_Name'] = res_df.apply(lambda row: f"{row['Model']}\n({row['Feature_Count']} feats)", axis=1)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=res_df, x='Short_Name', y='R2', palette='magma')
        
        plt.title('不同模型與特徵組合的 R2 分數比較')
        plt.xlabel('實驗組合')
        plt.ylabel('R2 Score (越高越好)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/5_Model_Comparison.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"❌ 繪製比較圖失敗: {e}")

# ==========================================
# 4. 執行繪圖
# ==========================================
plot_feature_importance()
plot_hourly_trend()
plot_one_week_zoom()
plot_scatter()
plot_experiment_comparison()

print(f"\n🎉 所有圖表已產生並儲存於 '{OUTPUT_DIR}' 資料夾中！")
