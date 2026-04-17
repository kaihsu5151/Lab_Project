Lab_Project

目前主要使用 mRNA、CNV、Methylation、miRNA 四種資料來訓練模型，
並比較不同特徵選法、tissue marker 去除方式，以及最後的多組學融合效果。

-----------------------------------
一、專題主題
-----------------------------------

1. 用不同組學資料分別訓練藥物預測模型
2. 比較哪些特徵選法效果比較好
3. 降低 tissue marker 對結果的干擾
4. 把不同組學的預測結果做 ensemble，提升整體表現

-----------------------------------
二、主要腳本
-----------------------------------

1. mRNA
- gpu_train_foldmap_ablation_BC_Bprime.py

2. CNV
- cnv_train_foldmap.py
- cnv_train_foldmap_refill.py

3. Methylation
- methylation_train.py
- methylation_train_refill.py

4. miRNA
- mirna_train_foldmap_fixed.py

5. Ensemble
- ensemble_stack_4omics.py

-----------------------------------
三、常用模式
-----------------------------------

(1) baseline
直接用 top-N 相關特徵訓練

(2) drop_nofill
先選 top-N，再把 tissue marker 去掉，不補回來

(3) drop_refill
先選特徵後去掉 tissue marker，再從後面的特徵補回來

(4) only_markers
只用 tissue marker 來訓練

-----------------------------------
四、基本流程
-----------------------------------

1. 先用其中一種組學訓練，產生 fold_map
2. 其他組學用同一份 fold_map 來重跑
3. 比較不同組學的表現
4. 把各組學 OOF prediction 丟進 ensemble
5. 看整體表現有沒有提升

-----------------------------------
五、常見輸出
-----------------------------------

訓練後通常會產生：

- 每個 drug 的預測結果
- 每個 fold 的指標
- fold_map.csv
- all_predictions.csv
- summary.csv
- scatter plot
- tissue marker table
- ensemble summary

-----------------------------------
六、使用到的工具
-----------------------------------

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

-----------------------------------
七、備註
-----------------------------------

- 藥物名稱與 cell line 名稱在腳本中會先做標準化
- 大部分模型都是以 per-drug 的方式訓練
- ensemble 使用的是 OOF prediction，避免資料洩漏
- 若資料對不起來，通常要先檢查 mapping、folds_root、drug 名稱與 cell line 名稱
