import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import platform
import xgboost as xgb
import csv
import os
from datetime import datetime

# 1. 初始設定
# ---------------------------------------------------------
warnings.filterwarnings('ignore')

# 設定中文字型 (根據作業系統自動選擇，避免亂碼)
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif system_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:  # Linux / Colab
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

plt.rcParams['axes.unicode_minus'] = False


# 2. 資料處理函數
# ---------------------------------------------------------
def load_and_preprocess_data(filepath):
    """
    讀取並預處理資料
    """
    df = pd.read_csv(filepath)
    
    # 確保時間欄位是 datetime 格式
    df['rent_time'] = pd.to_datetime(df['rent_time'])

    # X = df.drop(['rent_count', 'rent_time'], axis=1)
    # X = pd.get_dummies(X, columns=['rent_station', 'sarea'], drop_first=True)
    X=df[['hour', 'weekday', 'month', 'rent_station']]
    # X = pd.get_dummies(X, columns=['rent_station', 'sarea'], drop_first=True)
    X = pd.get_dummies(X, columns=['rent_station'], drop_first=True)
    y = df['rent_count']
    
    print(f"✅ 資料處理完成。樣本數: {X.shape[0]}, 特徵數: {X.shape[1]}")
    return X, y, df['rent_time'], X.columns


# 3. 訓練與評估函數
# ---------------------------------------------------------
def save_results(experiment_id, model_name, mae, rmse, r2, feature_names, filename="experiment_results.csv"):
    """
    將實驗結果儲存至 CSV 檔案
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 修改：新增 Experiment_ID 欄位
        if not file_exists:
            writer.writerow(['Experiment_ID', 'Timestamp', 'Model', 'MAE', 'RMSE', 'R2', 'Features'])
        
        features_str = "; ".join(map(str, feature_names))
        # 修改：寫入 experiment_id
        writer.writerow([experiment_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}", features_str])
    print(f"✅ 結果已儲存至 {filename} (ID: {experiment_id})")

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, experiment_id):
    """
    訓練模型並計算評估指標
    """
    print(f"\n🔄 正在訓練 {model_name} ...")
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 處理負值 (借車數不為負)
    y_pred = np.maximum(y_pred, 0)
    
    # 計算指標
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 {model_name} 評估結果:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R2   : {r2:.4f}")
    
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else []
    # 修改：傳入 experiment_id
    save_results(experiment_id, model_name, mae, rmse, r2, feature_names)

    return y_pred, model


# 4. 繪圖函數
# ---------------------------------------------------------
def get_feature_str(feature_names):
    """
    根據特徵名稱列表產生檔名後綴
    """
    # 移除可能導致檔名非法的字元
    clean_names = [str(f).replace(':', '').replace('/', '') for f in feature_names]
    
    full_str = "_".join(clean_names)
    # 如果檔名太長 (例如用了 One-Hot Encoding)，則簡化顯示特徵數量
    if len(full_str) > 50:
        return f"{len(feature_names)}_Features"
    return full_str

def save_plot(filename, folder):
    """
    儲存圖表到指定資料夾
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    plt.savefig(path)
    print(f"💾 圖表已儲存: {path}")

def plot_predictions(y_test, predictions_dict, time_index, feature_str, folder):
    """
    繪製時間序列預測對比圖 (最後 100 筆)
    """
    plt.figure(figsize=(14, 6))
    
    subset_n = 200
    if len(y_test) < subset_n:
        subset_n = len(y_test)
        
    subset_y_test = y_test[-subset_n:].values
    
    plt.plot(subset_y_test, label='實際值 (Actual)', color='black', linewidth=2, linestyle='--')
    
    colors = {'Linear Regression': '#1f77b4', 'Random Forest': '#2ca02c', 'XGBoost': '#d62728'}
    
    for name, y_pred in predictions_dict.items():
        subset_y_pred = y_pred[-subset_n:]
        color = colors.get(name, 'orange')
        plt.plot(subset_y_pred, label=f'{name} 預測', color=color, alpha=0.8)

    plt.title(f'模型預測結果對比 (最後 {subset_n} 筆資料)')
    plt.xlabel('時間順序')
    plt.ylabel('借車數量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖表
    save_plot(f"Predictions_{feature_str}.png", folder)
    plt.show()

def plot_feature_importance(model, feature_names, model_name, feature_str, folder):
    """
    繪製特徵重要性 (僅適用於樹模型)
    """
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 只顯示前 15 個重要特徵，若特徵不足 15 個則顯示全部
    top_n = min(15, len(importances))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name} - 前 {top_n} 重要特徵')
    plt.bar(range(top_n), importances[top_indices], align='center', color='skyblue')
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.xlabel('特徵名稱')
    plt.ylabel('重要性分數')
    plt.tight_layout()
    
    # 儲存圖表
    save_plot(f"FeatureImportance_{model_name}_{feature_str}.png", folder)
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name, feature_str, folder):
    """
    繪製 真實值 vs 預測值 的散佈圖
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    
    # 畫出完美的 45 度對角線
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'r--', label='完美預測線')
    
    plt.title(f'{model_name}: 真實值 vs 預測值')
    plt.xlabel('真實借車數 (Actual)')
    plt.ylabel('預測借車數 (Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖表
    save_plot(f"Scatter_{model_name}_{feature_str}.png", folder)
    plt.show()

def plot_hourly_comparison(y_test, y_pred, time_index, model_name, feature_str, folder):
    """
    繪製 平均小時趨勢圖 (檢查是否抓到早晚高峰)
    """
    # 建立一個臨時 DataFrame 來方便計算平均
    df_temp = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Hour': time_index.dt.hour
    })
    
    # 依小時分組計算平均值
    hourly_avg = df_temp.groupby('Hour').mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg.index, hourly_avg['Actual'], 'o-', label='真實平均', color='black', linewidth=2)
    plt.plot(hourly_avg.index, hourly_avg['Predicted'], 'o--', label='預測平均', color='red', linewidth=2)
    
    plt.title(f'{model_name}: 平均每小時借車量趨勢')
    plt.xlabel('小時 (0-23)')
    plt.ylabel('平均借車數')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖表
    save_plot(f"HourlyTrend_{model_name}_{feature_str}.png", folder)
    plt.show()

def plot_residuals_histogram(y_test, y_pred, model_name, feature_str, folder):
    """
    繪製 殘差 (誤差) 分佈直方圖
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title(f'{model_name}: 殘差分佈 (Residuals)')
    plt.xlabel('誤差值 (真實 - 預測)')
    plt.ylabel('頻率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖表
    save_plot(f"Residuals_{model_name}_{feature_str}.png", folder)
    plt.show()


# 5. 主程式
# ---------------------------------------------------------
if __name__ == "__main__":
    # 檔案名稱設定
    FILENAME = 'FINAL_MODEL_DATA_CLEAN.csv'
    
    # 載入資料
    X, y, time_col, feature_names = load_and_preprocess_data(FILENAME)
    
    # 產生用於檔名的特徵字串
    feature_str = get_feature_str(feature_names)
    
    # 修改：建立唯一的實驗編號 (ID)
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 建立本次實驗的專屬資料夾 (ID + 特徵簡稱)
    experiment_folder = os.path.join("results", f"{experiment_id}_{feature_str}")
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        
    print(f"🆔 本次實驗編號: {experiment_id}")
    print(f"📝 本次實驗特徵標籤: {feature_str}")
    print(f"📂 實驗結果將儲存於: {experiment_folder}")
    
    if X is not None:
        # 修改：指定 3 月份資料作為測試集 (Test Set)，其餘為訓練集
        print("ℹ️ 正在根據月份切分資料：3月為測試集...")
        
        # 建立 3 月份的遮罩 (Mask)
        test_mask = time_col.dt.month == 3
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        test_time = time_col[test_mask]
        
        print(f"訓練集 (非3月): {len(X_train)} 筆, 測試集 (3月): {len(X_test)} 筆")
        
        if len(X_test) == 0:
            print("❌ 警告：找不到 3 月份的資料！請檢查資料日期範圍。")
        
        predictions = {}
        models = {}

        # --- 模型 1: Linear Regression (線性回歸) ---
        lr = LinearRegression()
        # 修改：傳入 experiment_id
        pred_lr, model_lr = train_and_evaluate(lr, X_train, y_train, X_test, y_test, "Linear Regression", experiment_id)
        predictions['Linear Regression'] = pred_lr
        models['Linear Regression'] = model_lr

        # --- 模型 2: Random Forest (隨機森林) ---
        # n_estimators: 樹的數量, max_depth: 樹的最大深度 (避免過擬合)
        rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        # 修改：傳入 experiment_id
        pred_rf, model_rf = train_and_evaluate(rf, X_train, y_train, X_test, y_test, "Random Forest", experiment_id)
        predictions['Random Forest'] = pred_rf
        models['Random Forest'] = model_rf

        # --- 模型 3: XGBoost ---
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
        # 修改：傳入 experiment_id
        pred_xgb, model_xgb = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test, "XGBoost", experiment_id)
        predictions['XGBoost'] = pred_xgb
        models['XGBoost'] = model_xgb

        # --- 繪圖結果 ---
        print("\n📈 正在繪製預測對比圖...")
        plot_predictions(y_test, predictions, test_time, feature_str, experiment_folder)
        
        # --- 繪製隨機森林的特徵重要性 ---
        print("📊 正在繪製特徵重要性圖表...")
        plot_feature_importance(models['Random Forest'], feature_names, "Random Forest", feature_str, experiment_folder)
        
        plot_feature_importance(models['XGBoost'], feature_names, "XGBoost", feature_str, experiment_folder)
        
        # --- 繪製真實值 vs 預測值的散佈圖 ---
        for name, y_pred in predictions.items():
            plot_actual_vs_predicted(y_test, y_pred, name, feature_str, experiment_folder)
        
        # --- 繪製每小時的真實值與預測值趨勢 ---
        for name, y_pred in predictions.items():
            plot_hourly_comparison(y_test, y_pred, test_time, name, feature_str, experiment_folder)
        
        # --- 繪製殘差分佈直方圖 ---
        for name, y_pred in predictions.items():
            plot_residuals_histogram(y_test, y_pred, name, feature_str, experiment_folder)
             
        print("\n✅ 所有程式執行完畢。")