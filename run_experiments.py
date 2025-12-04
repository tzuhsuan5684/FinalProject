import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import csv
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 設定檔案路徑
DATA_FILE = 'FINAL_MODEL_DATA_CLEAN.csv'
RESULT_FILE = 'batch_experiment_results.csv'

def load_data(filepath):
    """讀取資料並轉換時間格式"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到資料檔案: {filepath}")
    
    df = pd.read_csv(filepath)
    df['rent_time'] = pd.to_datetime(df['rent_time'])
    return df

def get_features(df, feature_list):
    """
    根據特徵列表準備 X 矩陣，自動處理類別變數的 One-Hot Encoding
    """
    # 複製一份以免修改原始資料
    data = df.copy()
    
    # 檢查特徵是否存在於 DataFrame
    missing_cols = [col for col in feature_list if col not in data.columns]
    if missing_cols:
        raise ValueError(f"資料中缺少以下欄位: {missing_cols}")

    # 篩選需要的欄位
    X = data[feature_list]
    
    # 定義需要 One-Hot Encoding 的類別欄位
    categorical_cols = ['rent_station', 'sarea']
    
    # 找出目前特徵列表中包含的類別欄位
    cols_to_encode = [col for col in feature_list if col in categorical_cols]
    
    # 進行 One-Hot Encoding
    if cols_to_encode:
        X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
        
    return X

def save_result(experiment_id, model_name, features, mae, rmse, r2, note=""):
    """將單次實驗結果寫入 CSV"""
    file_exists = os.path.isfile(RESULT_FILE)
    
    with open(RESULT_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 寫入標題
        if not file_exists:
            writer.writerow(['Experiment_ID', 'Timestamp', 'Model', 'Features', 'MAE', 'RMSE', 'R2', 'Note'])
        
        feature_str = ";".join(features)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        writer.writerow([experiment_id, timestamp, model_name, feature_str, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}", note])
    
    print(f"✅ [Saved] {model_name} | Features: {len(features)} | R2: {r2:.4f}")

def run_experiment():
    print(f"🚀 開始批次實驗腳本...")
    
    # 1. 載入資料
    try:
        df = load_data(DATA_FILE)
        print(f"📚 資料載入成功，共 {len(df)} 筆資料")
    except Exception as e:
        print(f"❌ 資料載入失敗: {e}")
        return

    # 2. 定義實驗配置 (您可以隨時在此添加新的組合)
    # -----------------------------------------------------
    
    # (A) 特徵組合列表
    feature_sets = [
        # --- 1. 基礎時間特徵 (Baseline) ---
        ['hour', 'weekday', 'is_weekend', 'is_peak'],

        # --- 2. 時間 + 氣候特徵 (Dynamic) ---
        ['hour', 'weekday', 'temperature', 'rainfall', 'wind_speed'],

        # --- 3. 時間 + 站點靜態特徵 (Static - Location) ---
        # 這些特徵描述了站點的「屬性」，比單純用 rent_station ID 更具推廣性
        ['hour', 'weekday', 'latitude', 'longitude', 'Quantity'],  # 經緯度與車柱數

        # --- 4. 時間 + 周邊環境特徵 (Static - Environment) ---
        # 捷運、學校、公園、人口、商圈
        ['hour', 'weekday', 'mrt_count_800m', 'mrt_dist_nearest_m'],
        ['hour', 'weekday', 'school_count_800m', 'school_dist_nearest_m'],
        ['hour', 'weekday', 'park_count_800m', 'park_dist_nearest_m'],
        ['hour', 'weekday', 'population_count', 'shopping_district_count'],

        # --- 5. 綜合靜態特徵 (All Static) ---
        ['hour', 'weekday', 'Quantity', 'mrt_dist_nearest_m', 'school_dist_nearest_m', 'park_dist_nearest_m', 'population_count'],

        # --- 6. 全特徵 (All In) ---
        ['hour', 'weekday', 'month', 'is_weekend', 'is_peak', 
         'temperature', 'rainfall', 'wind_speed', 
         'Quantity', 'mrt_dist_nearest_m', 'school_dist_nearest_m', 'park_dist_nearest_m', 'population_count']
    ]
    
    # (B) 模型列表
    models = [
        ('LinearRegression', LinearRegression()),
        
        ('RandomForest_Depth5', RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)),
        ('RandomForest_Depth10', RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)),
        
        ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # -----------------------------------------------------

    # 產生本次批次實驗的 ID
    experiment_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 Batch ID: {experiment_batch_id}")

    # 3. 切分訓練與測試集 (固定策略：用 3 月做測試)
    # 這樣可以確保所有模型都在相同的基準上比較
    print("✂️  正在切分資料 (3月為測試集)...")
    is_test = df['rent_time'].dt.month == 3
    y_train = df.loc[~is_test, 'rent_count']
    y_test = df.loc[is_test, 'rent_count']
    
    print(f"   Train: {len(y_train)}, Test: {len(y_test)}")

    # 4. 迴圈執行實驗
    total_experiments = len(feature_sets) * len(models)
    current_count = 0

    for features in feature_sets:
        print(f"\n📦 Testing Feature Set: {features}")
        
        try:
            # 準備特徵矩陣 (包含 One-Hot Encoding)
            X = get_features(df, features)
            X_train = X[~is_test]
            X_test = X[is_test]
            
            print(f"   Feature Matrix Shape: {X.shape}")
            
            for model_name, model in models:
                current_count += 1
                print(f"   [{current_count}/{total_experiments}] Running {model_name}...", end="\r")
                
                # 訓練
                model.fit(X_train, y_train)
                
                # 預測
                y_pred = model.predict(X_test)
                y_pred = np.maximum(y_pred, 0) # 修正負值
                
                # 評估
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # 儲存結果
                save_result(experiment_batch_id, model_name, features, mae, rmse, r2, note="Month 3 Test")
                
        except Exception as e:
            print(f"\n❌ Error with features {features}: {str(e)}")

    print(f"\n🏁 所有實驗完成！結果已存至 {RESULT_FILE}")

if __name__ == "__main__":
    run_experiment()
