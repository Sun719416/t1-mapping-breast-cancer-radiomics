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



test_size     = 0.3
random_state  = 36


vt_threshold      = 1.0
pre_lasso_mode    = "rf"    # "rf" or "mrmr"
pre_lasso_fea_num = 100
icc_threshold     = 0.75    

train_ids_file   = r"C:\Users\Sun\Desktop\zhang_malignant_nii\train_regular.txt"
test_ids_file    = r"C:\Users\Sun\Desktop\zhang_malignant_nii\test_regular.txt"
icc_passed_file  = "icc_passed_regular.txt"


excel_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
files_r1 = {
    "DCE":      os.path.join(excel_dir, "DCE.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC.xlsx"),
}

# -----------------------------------------------------------------------------
def standardize_df(df, transformer=StandardScaler()):
    arr = transformer.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

df0 = pd.read_excel(files_r1["DCE"])
#df0 = pd.read_excel(files_r1["MAP_PRE"])
label_df= df0[["PatientID","Label"]].drop_duplicates().set_index("PatientID")

dfs_r1 = []
for seq, path in files_r1.items():
    df = pd.read_excel(path)

    drop_cols = ["Label","Sequence"]
    df = df.drop(columns=drop_cols)

    df = df.set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r1.append(df)

ro_all       = pd.concat(dfs_r1, axis=1, join="inner")
df_target    = label_df.loc[ro_all.index,"Label"].astype(int)


ro_all.to_excel(r"C:\Users\Sun\Desktop\zhang_malignant_nii\ro_all_merged_regular.xlsx", index=True)


all_ids      = ro_all.index.to_list()
train_ids, test_ids = train_test_split(
    all_ids, test_size=test_size,
    random_state=random_state,
    stratify=df_target
)

with open(train_ids_file, "w") as f:
    f.write("\n".join(train_ids))  

with open(test_ids_file, "w") as f:
    f.write("\n".join(test_ids)) 

ro_train = ro_all.loc[train_ids]
y_train  = df_target.loc[train_ids]
ro_test  = ro_all.loc[test_ids]
y_test   = df_target.loc[test_ids]


excel_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
files_r1 = {
    "DCE":      os.path.join(excel_dir, "DCE.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC.xlsx"),
}
files_r2 = {
    "DCE":      os.path.join(excel_dir, "DCE_reader2.xlsx"),
    "STIR":     os.path.join(excel_dir, "stir_reader2.xlsx"),
    "ADC":      os.path.join(excel_dir, "ADC_reader2.xlsx"),
}

icc_passed_file = "icc_passed_regular.txt"
icc_threshold = 0.75

dfs_r1 = []
for seq, path in files_r1.items():
    df = pd.read_excel(path)
    drop_cols = ["Label", "Sequence"]
    df = df.drop(columns=drop_cols).set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r1.append(df)

ro_all = pd.concat(dfs_r1, axis=1, join="inner")
ro_train = ro_all  

dfs_r2 = []
for seq, path in files_r2.items():
    df = pd.read_excel(path)
    drop_cols = ["Label", "Sequence"]
    df = df.drop(columns=drop_cols).set_index("PatientID")
    df.columns = [f"{col}_{seq}" for col in df.columns]
    dfs_r2.append(df)

ro2_all = pd.concat(dfs_r2, axis=1, join="inner")

reader2_patient_ids = ro2_all.index 
ro_train_filtered = ro_train.loc[ro_train.index.intersection(reader2_patient_ids)]  
ro2_train_filtered = ro2_all.loc[ro2_all.index.intersection(reader2_patient_ids)]  

ro_train_filtered = ro_train_filtered.loc[:, (ro_train_filtered != 0).any(axis=0)]

ro2_train_filtered = ro2_train_filtered.loc[:, (ro2_train_filtered != 0).any(axis=0)]


if "reader" in ro_train_filtered.columns:
    ro_train_filtered = ro_train_filtered.drop(columns=["reader"])
ro_train_filtered.insert(0, "reader", 1)  
if "target" in ro_train_filtered.columns:
    ro_train_filtered = ro_train_filtered.drop(columns=["target"])
ro_train_filtered.insert(1, "target", ro_train_filtered.index)  

if "reader" in ro2_train_filtered.columns:
    ro2_train_filtered = ro2_train_filtered.drop(columns=["reader"])
ro2_train_filtered.insert(0, "reader", 2)  

if "target" in ro2_train_filtered.columns:
    ro2_train_filtered = ro2_train_filtered.drop(columns=["target"])
ro2_train_filtered.insert(1, "target", ro2_train_filtered.index)  


data = pd.concat([ro_train_filtered, ro2_train_filtered], ignore_index=True)

feature_cols = data.columns.difference(["reader", "target", "PatientID"])

data_cleaned = data.dropna(subset=feature_cols)  

feature_cols = feature_cols[data_cleaned[feature_cols].var() > 0]


feature_cols = feature_cols[data_cleaned[feature_cols].nunique() > 1]  

icc_dict = {}
for col in feature_cols:
    icc_res = pg.intraclass_corr(
        data=data,
        targets="target",   
        raters="reader",    
        ratings=col,       
        nan_policy="omit"
    )
    icc_dict[col] = icc_res

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


print("Top 10 features by ICC(2,1):")
print(summary_icc21.head(10).to_string(index=False))

features_to_keep_regular = summary_icc21[summary_icc21["ICC"] >= 0.75]["Feature"].values


txt_file_path = os.path.join(excel_dir, "features_to_keep_regular.txt")
with open(txt_file_path, 'w') as f:
    for feature in features_to_keep_regular:
        f.write(f"{feature}\n")

print(f"\n筛选后的特征已保存到: {txt_file_path}")


data_1_cleaned = ro2_train_filtered[["reader", "target"] + list(features_to_keep_regular)]





import pandas as pd


file_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\ro_all_merged_regular.xlsx"  
features_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\features_to_keep_regular.txt"  


df = pd.read_excel(file_path, index_col="PatientID") 


with open(features_path, 'r') as file:
    selected_features = file.readlines()
selected_features = [feature.strip() for feature in selected_features]  


X_selected = df[selected_features] 


output_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\ICC_cleaned_regular.xlsx"
X_selected.to_excel(output_path, index=True)  

print(f"筛选后的数据已保存到 '{output_path}'.")



from scipy.stats import ttest_ind, levene
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

base_dir = r"C:\Users\Sun\Desktop\zhang_malignant_nii"
features_path = os.path.join(base_dir, "ICC_cleaned_regular.xlsx")  
train_path = os.path.join(base_dir, "train_regular.txt") 
test_path = os.path.join(base_dir, "test_regular.txt")  
label_path = os.path.join(base_dir, "patient-label.xlsx")  

df_features = pd.read_excel(features_path, index_col="PatientID")

with open(train_path, 'r') as file:
    train_patients = file.readlines()
train_patients = [patient.strip() for patient in train_patients]

with open(test_path, 'r') as file:
    test_patients = file.readlines()
test_patients = [patient.strip() for patient in test_patients]


df_labels = pd.read_excel(label_path)


df_labels['PatientID'] = df_labels['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()
df_features.index = df_features.index.str.replace(r'\s+', '', regex=True).str.lower()  

y_train = df_labels[df_labels['PatientID'].isin(train_patients)]["Label"]
y_test = df_labels[df_labels['PatientID'].isin(test_patients)]["Label"]

X_train = df_features.loc[train_patients]  
X_test = df_features.loc[test_patients]  


y_train.index = df_labels[df_labels['PatientID'].isin(train_patients)]['PatientID'].str.lower()
y_test.index = df_labels[df_labels['PatientID'].isin(test_patients)]['PatientID'].str.lower()


y_train = y_train.loc[X_train.index]  
y_test = y_test.loc[X_test.index]  

from sklearn.metrics import roc_auc_score

redundancy_threshold = 0.75

def auc_strength(series, y):
    """以训练集为基准，计算单特征与标签的判别力；返回 max(AUC, 1-AUC)。"""
    s = pd.Series(series)
    if s.nunique(dropna=True) <= 1:
        return 0.5
    s = s.fillna(s.median()) 
    try:
        auc = roc_auc_score(y, s)
        return max(auc, 1 - auc)
    except Exception:
        return abs(pd.Series(s).corr(pd.Series(y), method='pearson'))

strength = {col: auc_strength(X_train[col], y_train) for col in X_train.columns}

spearman_abs = X_train.corr(method='spearman').abs()

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

X_train = X_train[selected_feats]
X_test  = X_test[selected_feats]

redundancy_keep_path = os.path.join(base_dir, "redundancy_kept_features_regular+map.txt")
with open(redundancy_keep_path, "w") as f:
    for c in selected_feats:
        f.write(c + "\n")
print(f"冗余筛选后的特征列表已保存到: {redundancy_keep_path}")

scaler = StandardScaler()


X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


colNames = X_train_scaled.columns


ttest_selected = []
for colName in X_train_scaled.columns:

    if levene(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName])[1] > 0.05:
  
        if ttest_ind(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName])[1] < 0.1:
            ttest_selected.append(colName)
    else:
   
        if ttest_ind(X_train_scaled[y_train.values == 0][colName], X_train_scaled[y_train.values == 1][colName], equal_var=False)[1] < 0.1:
            ttest_selected.append(colName)


print(f"显著特征的数量: {len(ttest_selected)}")


print("显著特征列名:", ttest_selected)


X_train_selected_features = X_train_scaled[ttest_selected]


X_train_with_patientID = pd.DataFrame(X_train_selected_features, columns=X_train_selected_features.columns)
X_train_with_patientID['PatientID'] = X_train.index


X_train_with_patientID['Label'] = y_train.loc[train_patients].values  # 确保按顺序匹配

X_train_with_patientID.reset_index(drop=True, inplace=True)

X_train_with_patientID = X_train_with_patientID.sort_values(by='PatientID')

cols = ['PatientID', 'Label'] + [col for col in X_train_with_patientID.columns if col not in ['PatientID', 'Label']]
X_train_with_patientID = X_train_with_patientID[cols]

output_path_ttest_selected = r"C:\Users\Sun\Desktop\zhang_malignant_nii\X_train_selected_features_regular.xlsx"
X_train_with_patientID.to_excel(output_path_ttest_selected, index=False)



X_train_lasso = X_train_with_patientID.iloc[:, 2:].values 
y_train_lasso = X_train_with_patientID['Label'].values  


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


alphas=np.logspace(-3,2,50)
model_lassoCV=LassoCV(alphas=alphas,cv=10,max_iter=100000).fit(X_train_lasso, y_train_lasso)


MSEs = model_lassoCV.mse_path_  

MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

plt.figure()  
plt.errorbar(model_lassoCV.alphas_, MSEs_mean  
             , yerr=MSEs_std  
             , fmt='o'  
             , ms=3  
             , mfc='r'  
             , mec='r'  
             , ecolor='lightblue' 
             , elinewidth=2  
             , capsize=4 
             , capthick=1)
plt.semilogx() 
plt.axvline(model_lassoCV.alpha_, color='black', ls='--')  
plt.xlabel('Lamda')
plt.ylabel('MSE')
ax = plt.gca()
y_major_locator = ticker.MultipleLocator(0.05)  
ax.yaxis.set_major_locator(y_major_locator)
plt.show()

_, coefs, _ = model_lassoCV.path(X_train_lasso, y_train_lasso, alphas=alphas, max_iter=100000) 
plt.figure()
plt.semilogx(model_lassoCV.alphas_, coefs.T, '-')
plt.axvline(model_lassoCV.alpha_, color='black', ls='--')
plt.xlabel('Lamda')
plt.ylabel('Coefficient')
plt.show()


print(model_lassoCV.alpha_)
coef=pd.Series(model_lassoCV.coef_,index=ttest_selected)
print('Lasso picked '+str(sum(coef != 0))+' variables and eliminated the other '+str(sum(coef==0)))

selected_features = coef[coef != 0].index  

with open(r"C:\Users\Sun\Desktop\zhang_malignant_nii\lasso_selected_features_regular.txt", 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
print("Lasso selected features saved to 'lasso_selected_features_regular.txt'.")


X_train_final = pd.DataFrame(X_train_scaled[selected_features].values, columns=selected_features) 
final_data = data[['PatientID', 'Label']].join(X_train_final)  

output_path = r"C:\Users\Sun\Desktop\zhang_malignant_nii\lasso_selected_data_train_regular.xlsx"
final_data.to_excel(output_path, index=False)

print(f"Lasso selected data saved to '{output_path}'.")

X_train_final = X_train_scaled[selected_features].values
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features].values

X_train_final_df = pd.DataFrame(X_train_final, columns=selected_features)
X_test_final_df = pd.DataFrame(X_test_final, columns=selected_features)

print(X_train_final_df.head())
print(X_test_final_df.head())

