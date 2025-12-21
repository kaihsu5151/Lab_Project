import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定檔案路徑與參數
# ==========================================
# 資料夾路徑 (請修改成你的真實路徑)
data_dir = r'RawData'
result_dir = r'Results' # 結果存檔位置
os.makedirs(result_dir, exist_ok=True)

# 藥物檔案 (CSV)
drug_file = r'RawData\GDSC\final_drug_matrix.csv'

# 目標藥物
target_drug = "Selumetinib"

# ---【關鍵設定】特徵檔案清單 ---
# 只要把前面的 '#' 拿掉，程式就會自動讀取並加入分析
omics_files = {
    'mRNA': r'RawData\mRNA_final.csv',
    
    # --- 等你整理好資料，把下面這幾行的註解拿掉即可 ---
    # 'miRNA': r'C:\Project\RawData\miRNA_final.csv',
    # 'CNV': r'C:\Project\RawData\CNV_final.csv',
    # 'Methylation': r'C:\Project\RawData\Methylation_final.csv',
}

# ==========================================
# 2. 讀取藥物資料
# ==========================================
print(f"正在讀取藥物資料: {drug_file} ...")
if os.path.exists(drug_file):
    df_drug = pd.read_csv(drug_file, index_col=0)
else:
    print(f"❌ 找不到藥物檔案: {drug_file}")
    exit()

if target_drug not in df_drug.columns:
    print(f"❌ 藥物檔案中找不到 '{target_drug}'")
    exit()

# 取出目標藥物數據 (移除空值)
drug_series = df_drug[target_drug].dropna()
print(f"✅ 目標藥物 {target_drug} 共有 {len(drug_series)} 筆有效數據。")

# ==========================================
# 3. 迴圈分析各個特徵 (Omics)
# ==========================================
all_results = [] # 用來儲存所有特徵的 correlation 結果
summary_stats = [] # 用來儲存平均分數

print("\n🚀 開始特徵比較分析...")

for omics_name, file_path in omics_files.items():
    print(f"\n[{omics_name}] 正在分析...")
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到檔案: {file_path}，跳過。")
        continue
        
    # 讀取特徵資料
    print(f"  -> 讀取檔案中...")
    try:
        # 假設 index 是細胞株名稱
        df_omics = pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"  ❌ 讀取失敗: {e}")
        continue
        
    # 找交集細胞
    common_cells = df_omics.index.intersection(drug_series.index)
    print(f"  -> 交集細胞株數量: {len(common_cells)}")
    
    if len(common_cells) < 10:
        print(f"  ⚠️ 樣本數太少，跳過。")
        continue
        
    # 對齊資料
    X = df_omics.loc[common_cells]
    y = drug_series.loc[common_cells]
    
    # 計算 Correlation (Pearson)
    print(f"  -> 計算 {X.shape[1]} 個特徵的相關係數...")
    corrs = X.corrwith(y)
    
    # 取絕對值 (Abs Correlation)
    abs_corrs = corrs.abs()
    
    # 儲存結果 (為了畫 Boxplot)
    # 我們只存前 1000 個最強的特徵，避免電腦跑太慢，且更能代表該特徵的潛力
    # 如果你想存全部，就把 .head(1000) 拿掉
    top_corrs = abs_corrs.sort_values(ascending=False).head(1000)
    
    for val in top_corrs:
        all_results.append({
            'Omics Type': omics_name,
            'Abs Correlation': val
        })
        
    # 計算統計指標 (存表格用)
    stats = {
        'Omics Type': omics_name,
        'Mean |r| (Top 1000)': top_corrs.mean(),
        'Median |r| (Top 1000)': top_corrs.median(),
        'Max |r|': top_corrs.max()
    }
    summary_stats.append(stats)
    print(f"  ✅ {omics_name} 分析完成！ 平均強度: {stats['Mean |r| (Top 1000)']:.4f}")
    print(f"  ✅ {omics_name} 分析完成！ 最大強度: {stats['Max |r|']:.4f}")

# ==========================================
# 4. 畫圖比較 (Boxplot)
# ==========================================
if all_results:
    print("\n📊 正在繪製比較圖...")
    df_plot = pd.DataFrame(all_results)
    
    plt.figure(figsize=(10, 6))
    
    # 畫箱型圖
    sns.boxplot(data=df_plot, x='Omics Type', y='Abs Correlation', palette='Set2')
    
    # 加上一些裝飾
    plt.title(f'Feature Importance Comparison for {target_drug}\n(Top 1000 Correlated Features)', fontsize=14)
    plt.ylabel('Absolute Pearson Correlation (|r|)', fontsize=12)
    plt.xlabel('Omics Data Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 存圖
    save_path = os.path.join(result_dir, f'{target_drug}_Omics_Comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✨ 圖片已儲存: {save_path}")
    
    # 顯示統計表格
    df_stats = pd.DataFrame(summary_stats).set_index('Omics Type')
    print("\n🏆 各特徵集分數比較表:")
    print(df_stats)
    
    # 存表格
    stats_path = os.path.join(result_dir, f'{target_drug}_Omics_Stats.csv')
    df_stats.to_csv(stats_path)
    print(f"✨ 統計表已儲存: {stats_path}")

else:
    print("\n⚠️ 沒有產生任何結果，請檢查你的 mRNA_final.csv 路徑是否正確。")