import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import platform
import os
from datetime import datetime

# ==========================================
# 1. 初始設定
# ==========================================
warnings.filterwarnings('ignore')

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
# 2. 主程式
# ==========================================
if __name__ == "__main__":
    # 產生實驗編號（批次）
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 批次實驗編號: {batch_id}")
    print("=" * 60)
    
    # 載入資料
    print("📚 正在載入資料...")
    df = pd.read_csv('FINAL_MODEL_DATA_CLEAN.csv')
    df['rent_time'] = pd.to_datetime(df['rent_time'])
    
    # 準備特徵與目標
    features = ['hour', 'weekday', 'Quantity', 'mrt_dist_nearest_m', 
                'school_dist_nearest_m', 'park_dist_nearest_m', 'population_count']
    X = df[features]
    y = df['rent_count']
    
    # 切分訓練集與測試集 (3月為測試集)
    test_mask = df['rent_time'].dt.month == 3
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    test_time = df.loc[test_mask, 'rent_time']
    
    print(f"✅ 訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆\n")
    
    # 計算峰值門檻
    threshold = y_train.quantile(0.75)
    peak_count = np.sum(y_train > threshold)
    print(f"📊 峰值門檻 (Q3): {threshold:.1f}")
    print(f"📊 峰值樣本數: {peak_count} 筆 ({peak_count/len(y_train)*100:.1f}%)")
    print("=" * 60)
    
    # 測試不同的峰值權重
    peak_weights = [1.0, 2.0, 3.0, 4.0]
    all_results = []
    
    for peak_weight in peak_weights:
        print(f"\n🔄 測試峰值權重 = {peak_weight}")
        print("-" * 60)
        
        # 計算樣本權重
        sample_weights = np.where(y_train > threshold, peak_weight, 1.0)
        
        # 訓練模型
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # 預測與評估
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 計算峰值樣本的專屬指標
        peak_mask = y_test > threshold
        if peak_mask.sum() > 0:
            peak_mae = mean_absolute_error(y_test[peak_mask], y_pred[peak_mask])
            peak_rmse = np.sqrt(mean_squared_error(y_test[peak_mask], y_pred[peak_mask]))
        else:
            peak_mae = np.nan
            peak_rmse = np.nan
        
        print(f"   整體 MAE  : {mae:.4f}")
        print(f"   整體 RMSE : {rmse:.4f}")
        print(f"   整體 R²   : {r2:.4f}")
        print(f"   峰值 MAE  : {peak_mae:.4f}")
        print(f"   峰值 RMSE : {peak_rmse:.4f}")
        
        # 繪製每小時平均趨勢圖
        df_temp = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Hour': X_test['hour'].values
        })
        hourly_avg = df_temp.groupby('Hour').mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(hourly_avg.index, hourly_avg['Actual'], 'o-', label='真實平均', color='black', linewidth=2)
        plt.plot(hourly_avg.index, hourly_avg['Predicted'], 'o--', label='預測平均', color='red', linewidth=2)
        plt.xlabel('小時 (0-23)')
        plt.ylabel('平均借車數')
        plt.title(f'每小時平均借車量趨勢 (權重={peak_weight})')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 暫存圖表，稍後統一儲存
        hourly_fig_path = f'hourly_trend_weight_{peak_weight}.png'
        plt.savefig(hourly_fig_path, dpi=300)
        plt.close()
        
        # 儲存結果
        result = {
            'batch_id': batch_id,
            'peak_weight': peak_weight,
            'peak_threshold': threshold,
            'peak_samples_train': peak_count,
            'peak_samples_test': peak_mask.sum(),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'peak_MAE': peak_mae,
            'peak_RMSE': peak_rmse,
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_leaf': 1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        all_results.append(result)
    
    # ==========================================
    # 3. 儲存比較結果
    # ==========================================
    print("\n" + "=" * 60)
    print("💾 儲存實驗結果...")
    
    # 建立批次資料夾
    output_dir = os.path.join('results', f'batch_{batch_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 移動每小時趨勢圖到批次資料夾
    for peak_weight in peak_weights:
        temp_file = f'hourly_trend_weight_{peak_weight}.png'
        if os.path.exists(temp_file):
            final_path = os.path.join(output_dir, temp_file)
            os.rename(temp_file, final_path)
            print(f"✅ 已儲存: {final_path}")
    
    # 儲存比較結果
    results_df = pd.DataFrame(all_results)
    comparison_csv_path = f'{output_dir}/weight_comparison.csv'
    results_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 比較結果已儲存: {comparison_csv_path}")
    
    # 追加到總實驗紀錄
    all_experiments_path = 'results/all_experiments.csv'
    if os.path.exists(all_experiments_path):
        results_df.to_csv(all_experiments_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        results_df.to_csv(all_experiments_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已追加至總實驗紀錄: {all_experiments_path}")
    
    # ==========================================
    # 4. 繪製比較圖表
    # ==========================================
    print("\n📊 繪製比較圖表...")
    
    # 圖1: MAE 比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE 比較
    axes[0, 0].plot(results_df['peak_weight'], results_df['MAE'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('峰值權重')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('整體 MAE vs 峰值權重')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(peak_weights)
    
    # RMSE 比較
    axes[0, 1].plot(results_df['peak_weight'], results_df['RMSE'], 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('峰值權重')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('整體 RMSE vs 峰值權重')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(peak_weights)
    
    # R² 比較
    axes[1, 0].plot(results_df['peak_weight'], results_df['R2'], 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('峰值權重')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² vs 峰值權重')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(peak_weights)
    
    # 峰值 MAE 比較
    axes[1, 1].plot(results_df['peak_weight'], results_df['peak_MAE'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('峰值權重')
    axes[1, 1].set_ylabel('峰值 MAE')
    axes[1, 1].set_title('峰值樣本 MAE vs 峰值權重')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(peak_weights)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weight_comparison.png', dpi=300)
    plt.close()
    print(f"✅ 比較圖表已儲存: {output_dir}/weight_comparison.png")
    
    # ==========================================
    # 5. 顯示最佳結果
    # ==========================================
    print("\n" + "=" * 60)
    print("🏆 實驗結果總結")
    print("=" * 60)
    
    best_mae_idx = results_df['MAE'].idxmin()
    best_r2_idx = results_df['R2'].idxmax()
    best_peak_mae_idx = results_df['peak_MAE'].idxmin()
    
    print(f"\n✨ 最低 MAE: 權重 = {results_df.loc[best_mae_idx, 'peak_weight']}, MAE = {results_df.loc[best_mae_idx, 'MAE']:.4f}")
    print(f"✨ 最高 R²: 權重 = {results_df.loc[best_r2_idx, 'peak_weight']}, R² = {results_df.loc[best_r2_idx, 'R2']:.4f}")
    print(f"✨ 最低峰值 MAE: 權重 = {results_df.loc[best_peak_mae_idx, 'peak_weight']}, 峰值 MAE = {results_df.loc[best_peak_mae_idx, 'peak_MAE']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"🆔 批次實驗編號: {batch_id}")
    print(f"📁 結果資料夾: {output_dir}")
    print("🎉 執行完畢！")