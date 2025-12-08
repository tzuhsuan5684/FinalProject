import pandas as pd
import numpy as np
import os

def create_lag_features(df):
    """
    Creates lag features for rent_count (previous 3 hours and 24 hours).
    """
    print("正在建立滯後特徵 (Lag Features)...")
    df_lag = df.copy()
    
    # 確保時間格式正確
    if df_lag['rent_time'].dtype == 'object':
        df_lag['rent_time'] = pd.to_datetime(df_lag['rent_time'])
        
    # 必須先排序才能正確使用 shift
    df_lag = df_lag.sort_values(['rent_station', 'rent_time'])
    
    # Group by station to calculate lags
    # 前3小時
    df_lag['rent_count_lag_3'] = df_lag.groupby('rent_station')['rent_count'].shift(3)
    # 前24小時
    df_lag['rent_count_lag_24'] = df_lag.groupby('rent_station')['rent_count'].shift(24)
    
    # 填補 NaN (因為 shift 會產生空值)
    df_lag['rent_count_lag_3'] = df_lag['rent_count_lag_3'].fillna(0)
    df_lag['rent_count_lag_24'] = df_lag['rent_count_lag_24'].fillna(0)
    
    return df_lag

def create_interaction_features(df):
    """
    Creates interaction and polynomial features for the bike rental dataset.
    """
    df_new = df.copy()
    
    print("正在建立特徵組合...")


    # 3. Supply vs Demand Potential (供需潛力)
    # 人均站點容量
    df_new['capacity_per_pop'] = df_new['Quantity'] / (df_new['population_count'] + 1)
    
    # 4. Time Interactions (時間交互作用)
    # 週末尖峰
    df_new['weekend_peak'] = df_new['is_weekend'] * df_new['is_peak']
    
    # 工作日尖峰
    df_new['weekday_peak'] = (1 - df_new['is_weekend']) * df_new['is_peak']

    return df_new

def main():
    input_file = 'FINAL_MODEL_DATA_CLEAN.csv'
    output_file = 'FINAL_MODEL_DATA_WITH_FEATURES.csv'
    
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到輸入檔案 {input_file}")
        return

    print(f"正在讀取 {input_file}...")
    df = pd.read_csv(input_file)
    
    # 執行滯後特徵工程
    df = create_lag_features(df)
    
    # 執行特徵工程
    df_enhanced = create_interaction_features(df)
    
    print(f"原始特徵數量: {df.shape[1]}")
    print(f"新增後特徵數量: {df_enhanced.shape[1]}")
    
    print(f"正在儲存至 {output_file}...")
    df_enhanced.to_csv(output_file, index=False)
    print("完成！")

if __name__ == "__main__":
    main()
