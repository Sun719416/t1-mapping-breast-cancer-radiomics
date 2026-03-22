#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
#from mrmr import mrmr_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mannwhitneyu
import pingouin as pg
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


# ---------------------------- 超参 & 路径 ----------------------------
# 划分比例 & 随机种子
test_size     = 0.3
random_state  = 36

# 特征选择参数
vt_threshold      = 1.0
pre_lasso_mode    = "rf"    # "rf" or "mrmr"
pre_lasso_fea_num = 100
icc_threshold     = 0.75     # ICC 通过阈值

# 输出文件名
train_ids_file   = r"C:\Users\Sun\Desktop\zhang_malignant_nii\train_regular.txt"
test_ids_file    = r"C:\Users\Sun\Desktop\zhang_malignant_nii\test_regular.txt"
icc_passed_file  = "icc_passed_regular.txt"


# Reader1 的六个 Excel 文件路径
excel_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
files_r1 = {
    "DCE":      os.path.join(excel_dir, "DCE.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC.xlsx"),
    # "MAP_PRE":  os.path.join(excel_dir, "mappre.xlsx"),
    # "MAP_POST": os.path.join(excel_dir, "mappost.xlsx"),
    # "DELTA1":   os.path.join(excel_dir, "delta1.xlsx"),
    # "DELTA2":   os.path.join(excel_dir, "delta2.xlsx"),
}

# -----------------------------------------------------------------------------
def standardize_df(df, transformer=StandardScaler()):
    arr = transformer.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

# -----------------------------------------------------------------------------
# 1. 读取 & 合并 Reader1 特征
# -----------------------------------------------------------------------------
# 1.1 提取 PatientID–Label
df0 = pd.read_excel(files_r1["DCE"])
#df0 = pd.read_excel(files_r1["MAP_PRE"])
label_df= df0[["PatientID","Label"]].drop_duplicates().set_index("PatientID")

# 1.2 读取 6 个序列特征并重命名
dfs_r1 = []
for seq, path in files_r1.items():
    df = pd.read_excel(path)
    # 删除冗余列
    drop_cols = ["Label","Sequence"]
    df = df.drop(columns=drop_cols)
    # 索引 & 重命名
    df = df.set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r1.append(df)

ro_all       = pd.concat(dfs_r1, axis=1, join="inner")
df_target    = label_df.loc[ro_all.index,"Label"].astype(int)

# 输出合并后的数据表格到 Excel 文件
ro_all.to_excel(r"C:\Users\Sun\Desktop\zhang_malignant_nii\ro_all_merged_regular.xlsx", index=True)

# -----------------------------------------------------------------------------
# 2. 划分训练集 & 测试集，并保存 ID 列表
# -----------------------------------------------------------------------------
all_ids      = ro_all.index.to_list()
train_ids, test_ids = train_test_split(
    all_ids, test_size=test_size,
    random_state=random_state,
    stratify=df_target
)

with open(train_ids_file, "w") as f:
    f.write("\n".join(train_ids))  # 训练集 PatientID

with open(test_ids_file, "w") as f:
    f.write("\n".join(test_ids))   # 测试集 PatientID

ro_train = ro_all.loc[train_ids]
y_train  = df_target.loc[train_ids]
ro_test  = ro_all.loc[test_ids]
y_test   = df_target.loc[test_ids]


# In[5]:


# 已定义好的文件路径和参数
excel_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
files_r1 = {
    "DCE":      os.path.join(excel_dir, "DCE.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC.xlsx"),
    # "MAP_PRE":  os.path.join(excel_dir, "mappre.xlsx"),
    # "MAP_POST": os.path.join(excel_dir, "mappost.xlsx"),
    # "DELTA1":   os.path.join(excel_dir, "delta1.xlsx"),
    # "DELTA2":   os.path.join(excel_dir, "delta2.xlsx"),
}
files_r2 = {
    "DCE":      os.path.join(excel_dir, "DCE_reader2.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir_reader2.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC_reader2.xlsx"),
    # "MAP_PRE":  os.path.join(excel_dir, "mappre_reader2.xlsx"),
    # "MAP_POST": os.path.join(excel_dir, "mappost_reader2.xlsx"),
    # "DELTA1":   os.path.join(excel_dir, "delta1_reader2.xlsx"),
    # "DELTA2":   os.path.join(excel_dir, "delta2_reader2.xlsx"),
}

icc_passed_file = "icc_passed_regular.txt"
icc_threshold = 0.75

# 读取 Reader1 特征数据
dfs_r1 = []
for seq, path in files_r1.items():
    df = pd.read_excel(path)
    drop_cols = ["Label", "Sequence"]
    df = df.drop(columns=drop_cols).set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r1.append(df)

ro_all = pd.concat(dfs_r1, axis=1, join="inner")
ro_train = ro_all  # 这将是 Reader1 的特征数据，所有病人都在这里

# 读取 Reader2 特征数据
dfs_r2 = []
for seq, path in files_r2.items():
    df = pd.read_excel(path)
    drop_cols = ["Label", "Sequence"]
    df = df.drop(columns=drop_cols).set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r2.append(df)

ro2_all = pd.concat(dfs_r2, axis=1, join="inner")

# 只选择 train_ids 中存在的 PatientID
reader2_patient_ids = ro2_all.index  # Reader2 中的所有患者ID
ro_train_filtered = ro_train.loc[ro_train.index.intersection(reader2_patient_ids)]  # 只保留在 Reader2 中的病人
ro2_train_filtered = ro2_all.loc[ro2_all.index.intersection(reader2_patient_ids)]  # 只保留在 Reader2 中的病人

# -------- 1. 删除 Reader1 中值全为0的特征 --------
ro_train_filtered = ro_train_filtered.loc[:, (ro_train_filtered != 0).any(axis=0)]

# -------- 2. 删除 Reader2 中值全为0的特征 --------
ro2_train_filtered = ro2_train_filtered.loc[:, (ro2_train_filtered != 0).any(axis=0)]


# In[7]:


# -------- 1. 合并两个数据集 --------
# 检查是否已经有 'reader' 列，如果有则删除
if "reader" in ro_train_filtered.columns:
    ro_train_filtered = ro_train_filtered.drop(columns=["reader"])
ro_train_filtered.insert(0, "reader", 1)  # 标记为阅读者 1

if "target" in ro_train_filtered.columns:
    ro_train_filtered = ro_train_filtered.drop(columns=["target"])
ro_train_filtered.insert(1, "target", ro_train_filtered.index)  # 使用 PatientID 对齐

# 对 Reader2 数据做相同处理
if "reader" in ro2_train_filtered.columns:
    ro2_train_filtered = ro2_train_filtered.drop(columns=["reader"])
ro2_train_filtered.insert(0, "reader", 2)  # 标记为阅读者 2

if "target" in ro2_train_filtered.columns:
    ro2_train_filtered = ro2_train_filtered.drop(columns=["target"])
ro2_train_filtered.insert(1, "target", ro2_train_filtered.index)  # 使用 PatientID 对齐

# -------- 2. 合并数据 --------
data = pd.concat([ro_train_filtered, ro2_train_filtered], ignore_index=True)

# -------- 3. 提取特征列 --------
feature_cols = data.columns.difference(["reader", "target", "PatientID"])

# -------- 6. 去除包含缺失值的行 --------
data_cleaned = data.dropna(subset=feature_cols)  # 删除包含缺失值的行

# -------- 7. 去除没有变异的特征 --------
# 计算每个特征的方差，去掉方差为零的特征
feature_cols = feature_cols[data_cleaned[feature_cols].var() > 0]

# -------- 8. 去除特征值全为 1 的列 --------
feature_cols = feature_cols[data_cleaned[feature_cols].nunique() > 1]  # 删除全为 1 的列

# -------- 4. 计算 ICC --------
icc_dict = {}
for col in feature_cols:
    icc_res = pg.intraclass_corr(
        data=data,
        targets="target",   # 病人编号
        raters="reader",    # 阅读者编号
        ratings=col,        # 当前特征
        nan_policy="omit"
    )
    icc_dict[col] = icc_res

# -------- 5. 汇总 ICC 结果 --------
summary = (
    pd.concat({feat: df.set_index("Type") for feat, df in icc_dict.items()})
      .reset_index(level=0)
      .rename(columns={"level_0": "Feature"})
)

summary_icc21 = (
    summary.loc[summary.index == "ICC2", ["Feature", "ICC"]]
      .sort_values("ICC", ascending=False)
      .reset_index(drop=True)
)

# -------- 6. 输出和保存 --------
print("Top 10 features by ICC(2,1):")
print(summary_icc21.head(10).to_string(index=False))

# -------- 7. 筛选 ICC >= 0.75 的特征 --------
features_to_keep_regular = summary_icc21[summary_icc21["ICC"] >= 0.75]["Feature"].values

# 将这些特征名保存为 .txt 文件
txt_file_path = os.path.join(excel_dir, "features_to_keep_regular.txt")
with open(txt_file_path, 'w') as f:
    for feature in features_to_keep_regular:
        f.write(f"{feature}\n")

print(f"\n筛选后的特征已保存到: {txt_file_path}")

# -------- 8. 清理 data_2 --------
# 删除 ICC < 0.75 的特征
data_1_cleaned = ro2_train_filtered[["reader", "target"] + list(features_to_keep_regular)]


# In[9]:


import pandas as pd

# 定义文件路径
file_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\ro_all_merged_regular.xlsx"  # 输入数据文件
features_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\features_to_keep_regular.txt"  # 筛选特征名文件

# ========== Step 1: 读取 ro_all_merged.xlsx 数据 ==========
df = pd.read_excel(file_path, index_col="PatientID")  # 假设 PatientID 为索引列

# ========== Step 2: 读取 ICC 后筛选的特征名.txt ==========
with open(features_path, 'r') as file:
    selected_features = file.readlines()
selected_features = [feature.strip() for feature in selected_features]  # 去除换行符和空格

# ========== Step 3: 根据筛选的特征名筛选特征 ==========
# 只保留筛选的特征列
X_selected = df[selected_features]  # 保留特征列

# ========== Step 4: 输出筛选后的数据到 Excel 文件 ==========
output_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\ICC_cleaned_regular.xlsx"
X_selected.to_excel(output_path, index=True)  # 保留 PatientID 作为索引列

print(f"筛选后的数据已保存到 '{output_path}'.")


# In[11]:


from scipy.stats import ttest_ind, levene
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

base_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
features_path = os.path.join(base_dir, "ICC_cleaned_regular.xlsx")  # 包含所有病人特征数据，PatientID为索引
train_path = os.path.join(base_dir, "train_regular.txt")  # 包含训练集的 PatientID
test_path = os.path.join(base_dir, "test_regular.txt")  # 包含测试集的 PatientID
label_path = os.path.join(base_dir, "patient-label.xlsx")  # 包含病人ID和标签的表格，PatientID和Label列

# 读取特征数据（假设 PatientID 是索引）
df_features = pd.read_excel(features_path, index_col="PatientID")

# 读取训练集和测试集的 PatientID
with open(train_path, 'r') as file:
    train_patients = file.readlines()
train_patients = [patient.strip() for patient in train_patients]

with open(test_path, 'r') as file:
    test_patients = file.readlines()
test_patients = [patient.strip() for patient in test_patients]

# 读取标签数据（PatientID 和 Label）
df_labels = pd.read_excel(label_path)

# 去除 PatientID 中的空格，确保格式一致
df_labels['PatientID'] = df_labels['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()
df_features.index = df_features.index.str.replace(r'\s+', '', regex=True).str.lower()  # 确保 X_train 的索引也被清理

# 读取标签数据（PatientID 和 Label），确保标签和特征的 PatientID 一致
y_train = df_labels[df_labels['PatientID'].isin(train_patients)]["Label"]
y_test = df_labels[df_labels['PatientID'].isin(test_patients)]["Label"]

# 确保索引顺序一致
X_train = df_features.loc[train_patients]  # 根据 train_patients 获取训练集特征
X_test = df_features.loc[test_patients]  # 根据 test_patients 获取测试集特征

# 将 y_train 和 y_test 的索引设置为 PatientID，以便与 X_train 和 X_test 对齐
y_train.index = df_labels[df_labels['PatientID'].isin(train_patients)]['PatientID'].str.lower()
y_test.index = df_labels[df_labels['PatientID'].isin(test_patients)]['PatientID'].str.lower()

# 确保 y_train 和 y_test 使用相同的索引顺序
y_train = y_train.loc[X_train.index]  # 确保 y_train 的标签顺序与 X_train 一致
y_test = y_test.loc[X_test.index]  # 确保 y_test 的标签顺序与 X_test 一致

# ========== Step 3': 冗余分析（先于标准化与 t-test）==========
from sklearn.metrics import roc_auc_score

# 相关性阈值（常用 0.85~0.95；这里取 0.90）
redundancy_threshold = 0.75

def auc_strength(series, y):
    """以训练集为基准，计算单特征与标签的判别力；返回 max(AUC, 1-AUC)。"""
    s = pd.Series(series)
    if s.nunique(dropna=True) <= 1:
        return 0.5
    s = s.fillna(s.median())  # 仅用训练集的中位数填充
    try:
        auc = roc_auc_score(y, s)
        return max(auc, 1 - auc)
    except Exception:
        # 兜底：若 AUC 失败，用点双列相关的绝对值
        return abs(pd.Series(s).corr(pd.Series(y), method='pearson'))

# 1) 基于训练集计算每个特征的“强度”分数（判别力）
strength = {col: auc_strength(X_train[col], y_train) for col in X_train.columns}

# 2) 训练集特征间 Spearman 绝对相关矩阵
spearman_abs = X_train.corr(method='spearman').abs()

# 3) 按强度从高到低，贪心保留：与已选中任一特征 |ρ|>=阈值 的特征将被丢弃
ordered_feats = sorted(strength.keys(), key=lambda c: strength[c], reverse=True)
selected_feats = []
for f in ordered_feats:
    ok = True
    for s in selected_feats:
        rho = spearman_abs.loc[f, s]
        if pd.notna(rho) and rho >= redundancy_threshold:
            ok = False
            break
    if ok:
        selected_feats.append(f)

print(f"冗余分析：从 {X_train.shape[1]} 个特征保留 {len(selected_feats)} 个（|ρ|<{redundancy_threshold}）。")

# 4) 将冗余筛后的列应用到训练/测试集（避免信息泄漏：仅用训练集决策所选列）
X_train = X_train[selected_feats]
X_test  = X_test[selected_feats]

# （可选）保存冗余筛后的特征名
redundancy_keep_path = os.path.join(base_dir, "redundancy_kept_features_regular+map.txt")
with open(redundancy_keep_path, "w") as f:
    for c in selected_feats:
        f.write(c + "\n")
print(f"冗余筛选后的特征列表已保存到: {redundancy_keep_path}")

# ========== Step 4: 标准化特征 ==========
scaler = StandardScaler()

# 将标准化后的数据转回 DataFrame，并保留原始列名
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ========== Step 5: T-test 筛选特征 ==========
# 存储原始特征的列名
colNames = X_train_scaled.columns

# 执行统计检验，筛选出显著的特征
ttest_selected = []
for colName in X_train_scaled.columns:
    # Levene检验判断是否方差齐性
    if levene(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName])[1] > 0.05:
        # 若方差齐，使用独立样本t检验
        if ttest_ind(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName])[1] < 0.1:
            ttest_selected.append(colName)
    else:
        # 若方差不齐，使用Welch t检验
        if ttest_ind(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName], equal_var=False)[1] < 0.1:
            ttest_selected.append(colName)

# 打印选出的显著特征的数量
print(f"显著特征的数量: {len(ttest_selected)}")

# 打印选出的显著特征的列名
print("显著特征列名:", ttest_selected)

# 只保留 t-test 筛选的特征
X_train_selected_features = X_train_scaled[ttest_selected]

# 将 PatientID 列加入到标准化后的数据中
X_train_with_patientID = pd.DataFrame(X_train_selected_features, columns=X_train_selected_features.columns)
X_train_with_patientID['PatientID'] = X_train.index

# 使用 train_patients 来确保标签与 PatientID 一一对应
X_train_with_patientID['Label'] = y_train.loc[train_patients].values  # 确保按顺序匹配

# 重置索引，这样 PatientID 列不再是索引
X_train_with_patientID.reset_index(drop=True, inplace=True)

# 按照 PatientID 列升序排序
X_train_with_patientID = X_train_with_patientID.sort_values(by='PatientID')

# 将 PatientID 列放到最前面
cols = ['PatientID', 'Label'] + [col for col in X_train_with_patientID.columns if col not in ['PatientID', 'Label']]
X_train_with_patientID = X_train_with_patientID[cols]

# 保存到 Excel 文件
output_path_ttest_selected = r"C:\Users\Sun\Desktop\zhang_malignant_nii\X_train_selected_features_regular.xlsx"
X_train_with_patientID.to_excel(output_path_ttest_selected, index=False)


# In[13]:


# 直接使用原始数据，不进行打乱
X_train_lasso = X_train_with_patientID.iloc[:, 2:].values  # 特征矩阵（从第三列开始）
y_train_lasso = X_train_with_patientID['Label'].values  # 目标变量


# In[15]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#lasso特征筛选

alphas=np.logspace(-3,2,50)#参数可以自己调节选出最优alphas
model_lassoCV=LassoCV(alphas=alphas,cv=10,max_iter=100000).fit(X_train_lasso, y_train_lasso)


# In[17]:


# 作图1（MSE随lambda的变化）
MSEs = model_lassoCV.mse_path_  # ndarray of shape (n_alphas, n_folds)

# 计算每个lambda的MSE的mean和std
MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

plt.figure()  # 参数可选：dpi = 300
plt.errorbar(model_lassoCV.alphas_, MSEs_mean  # x和y
             , yerr=MSEs_std  # y error
             , fmt='o'  # 数据点标记
             , ms=3  # dot size
             , mfc='r'  # dot color
             , mec='r'  # dot margin color
             , ecolor='lightblue'  # error bar颜色
             , elinewidth=2  # error bar width
             , capsize=4  # cap length of error bar（error bar的帽）
             , capthick=1)
plt.semilogx()  # x轴log坐标
plt.axvline(model_lassoCV.alpha_, color='black', ls='--')  # axis vertical线，画在LASSOCV取的alpha处
plt.xlabel('Lamda')
plt.ylabel('MSE')
ax = plt.gca()
y_major_locator = ticker.MultipleLocator(0.05)  # y轴间隔
ax.yaxis.set_major_locator(y_major_locator)
plt.show()

# 作图2（LASSO path）
_, coefs, _ = model_lassoCV.path(X_train_lasso, y_train_lasso, alphas=alphas, max_iter=100000)  # coefs: ndarray of shape (n_features, n_alphas)
plt.figure()
plt.semilogx(model_lassoCV.alphas_, coefs.T, '-')
plt.axvline(model_lassoCV.alpha_, color='black', ls='--')
plt.xlabel('Lamda')
plt.ylabel('Coefficient')
plt.show()


# In[19]:


print(model_lassoCV.alpha_)
coef=pd.Series(model_lassoCV.coef_,index=ttest_selected)
print('Lasso picked '+str(sum(coef != 0))+' variables and eliminated the other '+str(sum(coef==0)))


# In[21]:


# 将保留的特征名保存为 .txt 文件
selected_features = coef[coef != 0].index  # 获取系数不为零的特征

with open(r"C:\Users\Sun\Desktop\zhang_malignant_nii\lasso_selected_features_regular.txt", 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
print("Lasso selected features saved to 'lasso_selected_features_regular.txt'.")

# 将保留的特征数据存入 .xlsx 文件，并保留 PatientID 和 Label 列
X_train_final = pd.DataFrame(X_train_scaled[selected_features].values, columns=selected_features)  # 转换为 DataFrame 并添加列名 # 选择保留的特征
final_data = data[['PatientID', 'Label']].join(X_train_final)  # 保留 PatientID 和 Label 列

# 输出合并后的数据到 Excel 文件
output_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\lasso_selected_data_train_regular.xlsx"
final_data.to_excel(output_path, index=False)

print(f"Lasso selected data saved to '{output_path}'.")


# In[23]:


X_train_final = X_train_scaled[selected_features].values
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features].values

# 将 NumPy 数组转换为 DataFrame 以便使用 head()
X_train_final_df = pd.DataFrame(X_train_final, columns=selected_features)
X_test_final_df = pd.DataFrame(X_test_final, columns=selected_features)

print(X_train_final_df.head())
print(X_test_final_df.head())

# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# 
