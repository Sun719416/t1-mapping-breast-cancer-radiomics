#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import os

# 文件路径
ro_all_merged_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ro_all_merged_regular+map.xlsx"  # 包含所有特征和病人数据
lasso_selected_features_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\lasso_selected_features_regular+map.txt" # Lasso选择的特征
train_set_txt_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\train_regular+map.txt"  # 训练集住院号
test_set_txt_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\test_regular+map.txt" # 测试集住院号
patient_label_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\patient_label.xlsx" # 标签文件

# 生成ROC曲线及AUC的函数
def gen_roc_curve(model_name, model, X_data, y_data, feature_set="delta", save_fig=False):
    print("Info: " + feature_set + "_" + model_name)
    y_pre = model.predict_proba(X_data)
    y_pre_pos = list(y_pre[:, 1])  # 正类的概率估计，和lr_model.predict(X_test)符合
    fpr, tpr, _ = roc_curve(y_data, y_pre_pos)
    auc = roc_auc_score(y_data, y_pre_pos)

    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name + ' ROC curve')
    plt.legend(loc='lower right')
    if save_fig:
        plt.savefig(os.path.join(r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_regular+map", feature_set + "_" + model_name))
    plt.show()

# 读取特征文件
with open(lasso_selected_features_file) as f:
    selected_list = f.readlines()
selected_list = [feature.strip() for feature in selected_list]

# 加载病人数据及所有特征
ro_all_merged = pd.read_excel(ro_all_merged_file, index_col="PatientID", converters={0: str})

# 加载标签数据
patient_label = pd.read_excel(patient_label_file, converters={0: str})
# 去掉 patient_label 中 PatientID 列的空格，并统一为小写或大写（统一格式）
patient_label['PatientID'] = patient_label['PatientID'].str.replace(" ", "").str.lower()

# 合并标签数据到特征数据中，并确保 PatientID 仍然是索引
df_full = pd.merge(ro_all_merged, patient_label[['PatientID', 'Label']], how='inner', on='PatientID')

# 确保 PatientID 是索引
df_full.set_index('PatientID', inplace=True)

# 读取训练集和测试集住院号
with open(train_set_txt_file) as f:
    train_set = f.readlines()
train_set = [n.strip() for n in train_set]

with open(test_set_txt_file) as f:
    test_set = f.readlines()
test_set = [n.strip() for n in test_set]

# 选择训练集和测试集数据
train_mask = df_full.index.isin(train_set)
test_mask = df_full.index.isin(test_set)

# 如果没有匹配数据，可以加上打印调试信息：
if train_mask.sum() == 0 or test_mask.sum() == 0:
    print("Warning: No matching patients found in train/test sets.")
    print(f"Train Mask: {train_mask}")
    print(f"Test Mask: {test_mask}")

X = df_full[selected_list]  # 使用选择的特征
y = df_full['Label']  # 目标值

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# 标准化数据
# def standardize_df(df, transformer=StandardScaler()):
#     """标准化df并转回df"""
#     transformed_array = transformer.fit_transform(df)
#     transformed_df = pd.DataFrame(transformed_array, columns=df.columns, index=df.index)
#     return transformed_df

# X_train = standardize_df(X_train)
# X_test = standardize_df(X_test)


# In[41]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, roc_curve, brier_score_loss, make_scorer
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# ====================== 工具函数 ======================

def find_best_youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    J = tpr - fpr
    best = np.where(J == J.max())[0]
    return float(np.median(thresholds[best])), float(J.max())


def evaluate_binary_classifier(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_true, y_prob)
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "AUC": auc,
        "Brier": brier,
        "Threshold": threshold
    }


def print_metrics_classic(train_metrics: dict, test_metrics: dict, best_params: dict, prefix: str = "LR"):
    print(f"\nBest parameters from cross-validation: {best_params}")
    print(f"Fixed threshold (Youden on Train CV): {final_threshold:.4f}")

    print(f"{prefix} Train Accuracy: {train_metrics.get('Accuracy', float('nan')):.4f}")
    print(f"{prefix} Test Accuracy: {test_metrics.get('Accuracy', float('nan')):.4f}")

    print(f"{prefix} Train Sensitivity (Recall): {train_metrics.get('Sensitivity', float('nan')):.4f}")
    print(f"{prefix} Test Sensitivity (Recall): {test_metrics.get('Sensitivity', float('nan')):.4f}")

    print(f"{prefix} Train Specificity: {train_metrics.get('Specificity', float('nan')):.4f}")
    print(f"{prefix} Test Specificity: {test_metrics.get('Specificity', float('nan')):.4f}")

    print(f"{prefix} Train AUC: {train_metrics.get('AUC', float('nan')):.3f}")
    print(f"{prefix} Test  AUC: {test_metrics.get('AUC', float('nan')):.3f}")

    print(f"{prefix} Train F1 Score:      {train_metrics.get('F1', float('nan')):.4f}")
    print(f"{prefix} Test  F1 Score:      {test_metrics.get('F1', float('nan')):.4f}")

    print(f"{prefix} Train Precision:     {train_metrics.get('Precision', float('nan')):.4f}")
    print(f"{prefix} Test  Precision:     {test_metrics.get('Precision', float('nan')):.4f}")

    print(f"{prefix} Train Brier Score:   {train_metrics.get('Brier', float('nan')):.4f}")
    print(f"{prefix} Test  Brier Score:   {test_metrics.get('Brier', float('nan')):.4f}")


def save_radiomics_score(path, index, scores, labels=None):
    df = pd.DataFrame({'PatientID': index, 'RadiomicsScore': scores})
    if labels is not None:
        labels_aligned = pd.Series(labels, index=index)
        df["Label"] = labels_aligned.values
    df.to_excel(path, index=False)
    print(f"Saved: {path}")
    return df


def plot_roc(y_true, y_prob, title, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_calibration(y_true, y_prob, name, save_path=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration Curve: {name}")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def bootstrap_metrics_fixed_model(model, X, y, threshold, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    probs = model.predict_proba(X)[:, 1]
    out = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y), len(y))
        y_b = y.iloc[idx]
        p_b = probs[idx]
        yhat = (p_b >= threshold).astype(int)
        if y_b.nunique() < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_b, yhat, labels=[0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        f1 = f1_score(y_b, yhat, zero_division=0)
        auc = roc_auc_score(y_b, p_b)
        out.append([auc, acc, rec, spec, f1])
    arr = np.array(out)
    means = arr.mean(axis=0)
    lows, highs = np.percentile(arr, [2.5, 97.5], axis=0)
    return means, np.vstack([lows, highs]).T


def cv_youden_threshold(X, y, best_params, n_splits=5, seed=42):
    """
    使用GridSearch找到的最佳参数，在训练集上进行CV来确定最稳定的阈值。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ths = []

    # 从 best_params 中提取参数
    C_val = best_params.get('lr__C', 1.0)
    penalty_val = best_params.get('lr__penalty', 'l2')
    solver_val = best_params.get('lr__solver', 'liblinear')
    class_weight_val = best_params.get('lr__class_weight', None)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        # 严格使用最佳参数构建临时模型
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty=penalty_val,
                solver=solver_val,
                max_iter=10000,
                class_weight=class_weight_val,
                C=C_val
            ))
        ])
        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]

        # 在该折训练子集上寻找最佳 Youden 阈值
        th, _ = find_best_youden_threshold(y_tr, p_tr)
        ths.append(th)

    return float(np.median(ths))


# ====================== 主流程 ======================

roc_dir = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_regular+map"
os.makedirs(roc_dir, exist_ok=True)

# 1. Pipeline 定义
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=10000))
])

# 2. 扩大超参数搜索空间
#    - 加入 penalty: 'l1' (Lasso) 可以进行特征筛选，可能减少过拟合
#    - 加入 class_weight: 'balanced' vs None，看哪种权重的效果更好
#    - 扩大 C 的范围
param_grid = [
    # 组合 1: 针对支持 L1 和 L2 的 solver (liblinear, saga)
    {
        "lr__solver": ["liblinear","saga"],
        "lr__penalty": ["l1", "l2"],
        "lr__C": np.logspace(-2, 0, 100),
        "lr__class_weight": ["balanced", None]
    },
    # 组合 2: 针对只支持 L2 的 solver (lbfgs) - 通常 lbfgs 速度快且效果不错
    {
        "lr__solver": ["lbfgs"],
        "lr__penalty": ["l2"], # lbfgs 不支持 l1
        "lr__C": np.logspace(-2, 0, 100),
        "lr__class_weight": ["balanced", None]
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_  # 获取最佳参数字典

# 3. 概率预测
y_train_prob = best_model.predict_proba(X_train)[:, 1]
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# 4. 计算阈值 (使用 CV + 最佳参数)
final_threshold = cv_youden_threshold(X_train, y_train, best_params)

# 5. 评估
train_metrics = evaluate_binary_classifier(y_train, y_train_prob, final_threshold)
test_metrics = evaluate_binary_classifier(y_test, y_test_prob, final_threshold)
print_metrics_classic(train_metrics, test_metrics, best_params, prefix="LR")

# Radiomics Score 保存
save_radiomics_score(
    r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_train_radiomics_scores_regular+map.xlsx",
    X_train.index, y_train_prob, labels=y_train
)
save_radiomics_score(
    r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_test_radiomics_scores_regular+map.xlsx",
    X_test.index, y_test_prob, labels=y_test
)

# ROC 曲线 & 校准曲线
plot_roc(y_train, y_train_prob, "LR Train ROC", os.path.join(roc_dir, "LR_train_ROC.png"))
plot_roc(y_test, y_test_prob, "LR Test ROC", os.path.join(roc_dir, "LR_test_ROC.png"))

plot_calibration(y_train, y_train_prob, "Train", os.path.join(roc_dir, "LR_train_calibration.png"))
plot_calibration(y_test, y_test_prob, "Test", os.path.join(roc_dir, "LR_test_calibration.png"))

# Bootstrapping CI
means_train, ci_train = bootstrap_metrics_fixed_model(best_model, X_train, y_train, final_threshold)
means_test, ci_test = bootstrap_metrics_fixed_model(best_model, X_test, y_test, final_threshold)

for name, means, ci in [("训练集", means_train, ci_train), ("测试集", means_test, ci_test)]:
    print(f"\nBootstrapping {name}:")
    for metric_name, mean, (low, high) in zip(["AUC", "Acc", "Recall", "Spec", "F1"], means, ci):
        print(f"{metric_name}: {mean:.3f} (95% CI: {low:.3f} ~ {high:.3f})")


# In[7]:


import sys
print("Python版本号:", sys.version)
print("\n版本详细信息:")
print(f"主版本: {sys.version_info.major}")
print(f"次版本: {sys.version_info.minor}")
print(f"微版本: {sys.version_info.micro}")
print(f"发布级别: {sys.version_info.releaselevel}")
print(f"序列号: {sys.version_info.serial}")

# 或者简洁版本
print(f"\n当前Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


# In[43]:


# ====================== Random Forest (Fβ阈值 + CV汇总阈值版) ======================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

# ---------------------- 你的数据 ----------------------
# X_train, X_test: pandas.DataFrame
# y_train, y_test: pandas.Series (index与X对应)

# ---------------------- 配置 ----------------------
RANDOM_STATE   = 42
BETA           = 2.0      # Fβ 中的 β；β>1 更偏向 Recall（例如 2.0）
TARGET_RECALL  = None     # 若希望 Recall≥某阈值，如0.70；无硬约束设为 None
CV_SPLITS      = 5
roc_dir        = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_regular+map"
save_dir_rf    = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\RF"
os.makedirs(roc_dir, exist_ok=True)
os.makedirs(save_dir_rf, exist_ok=True)

# ====================== 工具函数 ======================
from sklearn.metrics import fbeta_score

def find_best_threshold_by_metric(y_true, y_prob, beta=1.0, target_recall=None):
    """
    扫描“唯一概率阈值”（包含0和1端点），返回：
    - 最优阈值 t（float）
    - 对应的 Fβ 分数（float）
    - 当时的 Recall（float）
    说明：β>1 更看重 Recall；β<1 更看重 Precision；β=1 即 F1。
    若 target_recall 给出，则只在 Recall>=target 的候选中取 Fβ 最高者；
    若无候选满足，则回退到全局 Fβ 最高的阈值。
    """
    thr = np.r_[0.0, np.sort(np.unique(y_prob)), 1.0]
    best_t, best_score, best_rec = 0.5, -1.0, 0.0
    for t in thr:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if target_recall is not None and rec < target_recall:
            continue
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t, best_rec = score, t, rec
    if target_recall is not None and best_score < 0:  # 回退
        return find_best_threshold_by_metric(y_true, y_prob, beta=beta, target_recall=None)
    return float(best_t), float(best_score), float(best_rec)

def cv_threshold_by_metric(X, y, best_params, beta=1.0, target_recall=None, n_splits=5, seed=RANDOM_STATE):
    """
    K折：每折只用该折训练子集拟合RF，并在该子集上用 Fβ/Recall 策略定阈值；最后取“中位数”。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ths = []
    for tr_idx, _ in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        model = RandomForestClassifier(**best_params, random_state=seed)
        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]
        t, _, _ = find_best_threshold_by_metric(y_tr, p_tr, beta=beta, target_recall=target_recall)
        ths.append(t)
    return float(np.median(ths))

def evaluate_binary_classifier(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_true, y_prob)
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "AUC": auc,
        "Brier": brier,
        "Threshold": threshold
    }

def print_metrics_classic(train_metrics: dict, test_metrics: dict,
                          best_params: dict = None, final_threshold: float = None,
                          prefix: str = "RF"):
    if best_params is not None:
        print(f"Best params from cross-validation: {best_params}")
    if final_threshold is not None:
        print(f"Fixed threshold (F{BETA:g} via CV median): {final_threshold:.4f}")

    print(f"{prefix} Train Accuracy: {train_metrics.get('Accuracy', float('nan')):.4f}")
    print(f"{prefix} Test Accuracy: {test_metrics.get('Accuracy', float('nan')):.4f}")

    print(f"{prefix} Train Sensitivity (Recall): {train_metrics.get('Sensitivity', float('nan')):.4f}")
    print(f"{prefix} Test Sensitivity (Recall): {test_metrics.get('Sensitivity', float('nan')):.4f}")

    print(f"{prefix} Train Specificity: {train_metrics.get('Specificity', float('nan')):.4f}")
    print(f"{prefix} Test Specificity: {test_metrics.get('Specificity', float('nan')):.4f}")

    print(f"{prefix} Train AUC: {train_metrics.get('AUC', float('nan')):.3f}")
    print(f"{prefix} Test  AUC: {test_metrics.get('AUC', float('nan')):.3f}")

    print(f"{prefix} Train F1 Score:      {train_metrics.get('F1', float('nan')):.4f}")
    print(f"{prefix} Test  F1 Score:      {test_metrics.get('F1', float('nan')):.4f}")

    print(f"{prefix} Train Recall:        {train_metrics.get('Recall', float('nan')):.4f}")
    print(f"{prefix} Test  Recall:        {test_metrics.get('Recall', float('nan')):.4f}")

    print(f"{prefix} Train Precision:     {train_metrics.get('Precision', float('nan')):.4f}")
    print(f"{prefix} Test  Precision:     {test_metrics.get('Precision', float('nan')):.4f}")

    print(f"{prefix} Train Brier Score:   {train_metrics.get('Brier', float('nan')):.4f}")
    print(f"{prefix} Test  Brier Score:   {test_metrics.get('Brier', float('nan')):.4f}")

    BA_tr = 0.5 * (train_metrics.get("Sensitivity", np.nan) + train_metrics.get("Specificity", np.nan))
    BA_te = 0.5 * (test_metrics.get("Sensitivity", np.nan)  + test_metrics.get("Specificity", np.nan))
    J_tr  = train_metrics.get("Sensitivity", np.nan) + train_metrics.get("Specificity", np.nan) - 1.0
    J_te  = test_metrics.get("Sensitivity", np.nan)  + test_metrics.get("Specificity", np.nan) - 1.0
    print(f"{prefix} Train BalancedAcc:   {BA_tr:.4f}")
    print(f"{prefix} Test  BalancedAcc:   {BA_te:.4f}")
    print(f"{prefix} Train Youden Index:  {J_tr:.4f}")
    print(f"{prefix} Test  Youden Index:  {J_te:.4f}")

def save_radiomics_score(path, index, scores, labels=None, verbose=True):
    df = pd.DataFrame({'PatientID': index, 'RadiomicsScore': scores})
    if labels is not None:
        labels_aligned = pd.Series(labels).reindex(index)
        df["Label"] = labels_aligned.values
    df.to_excel(path, index=False)
    if verbose:
        print(f"Saved: {path}")
    return df

def plot_roc(y_true, y_prob, title, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_calibration(y_true, y_prob, name, save_path=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration Curve: {name}")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def bootstrap_metrics_fixed_model(model, X, y, threshold, n_iter=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    probs = model.predict_proba(X)[:, 1]
    out = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y), len(y))
        y_b = y.iloc[idx]
        p_b = probs[idx]
        yhat = (p_b >= threshold).astype(int)
        if y_b.nunique() < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_b, yhat, labels=[0,1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        f1  = f1_score(y_b, yhat, zero_division=0)
        auc = roc_auc_score(y_b, p_b)
        out.append([auc, acc, rec, spec, f1])
    arr = np.array(out)
    means = arr.mean(axis=0)
    lows, highs = np.percentile(arr, [2.5, 97.5], axis=0)
    return means, np.vstack([lows, highs]).T

def print_bootstrap_percentile(name, means, ci):
    for metric_name, mean, (low, high) in zip(["AUC","Acc","Recall","Spec","F1"], means, ci):
        print(f"{metric_name}: {mean:.3f} (95% CI: {low:.3f} ~ {high:.3f})")

# ====================== 模型选择（GridSearchCV, AUC评分） ======================
rfc = RandomForestClassifier(random_state=RANDOM_STATE)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 8, 12],
    "min_samples_split": [8, 12],
    "max_features": ["sqrt", 0.3],
    "max_samples": [0.6, 0.7, 0.8],
    "bootstrap": [True],
    #"criterion": ["gini"],
    "class_weight": ['balanced']
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(rfc, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
gs.fit(X_train, y_train)

best_params = gs.best_params_
rf_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)

# ====================== 概率 & 阈值（Fβ + CV汇总） ======================
y_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_test_prob  = rf_model.predict_proba(X_test)[:, 1]

final_threshold = cv_threshold_by_metric(
    X_train, y_train, best_params,
    beta=BETA, target_recall=TARGET_RECALL,
    n_splits=CV_SPLITS, seed=RANDOM_STATE
)

# ====================== 固定阈值评估 ======================
train_metrics = evaluate_binary_classifier(y_train, y_train_prob, final_threshold)
test_metrics  = evaluate_binary_classifier(y_test,  y_test_prob,  final_threshold)

print_metrics_classic(train_metrics, test_metrics,
                      best_params=best_params, final_threshold=final_threshold, prefix="RF")

# ====================== Radiomics Score 保存 ======================
train_radiomics_df = save_radiomics_score(
    os.path.join(save_dir_rf, "RF_train_radiomics_scores_regular+map.xlsx"),
    index=X_train.index, scores=y_train_prob, labels=y_train
)
test_radiomics_df = save_radiomics_score(
    os.path.join(save_dir_rf, "RF_test_radiomics_scores_regular+map.xlsx"),
    index=X_test.index,  scores=y_test_prob,  labels=y_test
)

print("\nTrain Radiomics Scores (head):")
print(train_radiomics_df.head())
print("\nTest Radiomics Scores (head):")
print(test_radiomics_df.head())
print("\nRadiomics Score 已保存！")

# ====================== ROC & 校准曲线 ======================
plot_roc(y_train, y_train_prob, "RF Train ROC", os.path.join(roc_dir, "RF_train_ROC.png"))
plot_roc(y_test,  y_test_prob,  "RF Test ROC",  os.path.join(roc_dir, "RF_test_ROC.png"))

plot_calibration(y_train, y_train_prob, "RF Train", os.path.join(roc_dir, "RF_train_calibration.png"))
plot_calibration(y_test,  y_test_prob, "RF Test",  os.path.join(roc_dir, "RF_test_calibration.png"))

# ====================== Bootstrap 百分位CI（固定模型+样本扰动） ======================
means_tr, ci_tr = bootstrap_metrics_fixed_model(rf_model, X_train, y_train, final_threshold, n_iter=500)
means_te, ci_te = bootstrap_metrics_fixed_model(rf_model, X_test,  y_test,  final_threshold, n_iter=500)

print("\nBootstrap summary (Train, percentile CI):")
print_bootstrap_percentile("Train", means_tr, ci_tr)
print("\nBootstrap summary (Test, percentile CI):")
print_bootstrap_percentile("Test", means_te, ci_te)


# In[25]:


#-----------------------------------------------------SHAP-------------------------------------------------------
import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# （可选）无交互环境下确保能写文件
# plt.switch_backend("Agg")

# 在 RF 文件夹下创建 SHAP_regular+map（按你之前的要求）
os.makedirs(r"C:\Users\Sun\Desktop\luminal_and_nonluminal\RF\SHAP_regular+map", exist_ok=True)

# 目标保存到 LR\SHAP_regular+map
shap_dir = r"C:\Users\Sun\Desktop\luminal_and_nonluminal\RF\SHAP_regular+map"
os.makedirs(shap_dir, exist_ok=True)

# 1) 创建解释器 & 计算 SHAP 值
explainer = shap.TreeExplainer(rf_model)
shap_values_train = explainer.shap_values(X_train)

# 兼容 list/np.array 两种返回；取正类
if isinstance(shap_values_train, list):
    shap_values_pos = shap_values_train[1]          # (n_samples, n_features)
else:
    # 兼容你环境里可能返回 (n, f, 2)
    shap_values_pos = shap_values_train[:, :, 1] if shap_values_train.ndim == 3 else shap_values_train

print("Shape(shap_values_pos):", np.array(shap_values_pos).shape)
print("Shape(X_train):", X_train.shape)

# 2) 重要度排序
mean_abs = np.abs(shap_values_pos).mean(axis=0).reshape(-1)
cols = np.asarray(X_train.columns)
assert mean_abs.shape[0] == cols.shape[0], "特征数不一致：SHAP 与 X_train 列数不同"
top_idx = np.argsort(mean_abs)[::-1]
top_features = cols[top_idx].tolist()

print("Top important features (前5):")
for name in top_features[:5]:
    print(" -", name)

def _save_current(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

def _safe_name(s):
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in str(s))

# 3) Summary（bar）
plt.figure()
shap.summary_plot(shap_values_pos, X_train, plot_type="bar", show=False)
_save_current(os.path.join(shap_dir, "RF_SHAP_summary_bar_regular+map.png"))

# 4) Summary（beeswarm）
plt.figure()
shap.summary_plot(shap_values_pos, X_train, show=False)
_save_current(os.path.join(shap_dir, "RF_SHAP_summary_beeswarm_regular+map.png"))

# 5) Dependence（前5重要特征）
for feat in top_features[:5]:
    plt.figure()
    shap.dependence_plot(str(feat), shap_values_pos, X_train, show=False)
    _save_current(os.path.join(shap_dir, f"RF_SHAP_dependence_{_safe_name(feat)}_regular+map.png"))

# 6) Force plot（示例保存第 1 个样本）
i = 0
plt.figure()
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
    shap_values_pos[i, :],
    X_train.iloc[i, :],
    matplotlib=True,  # 关键：启用 matplotlib 渲染
    show=False
)
_save_current(os.path.join(shap_dir, f"RF_SHAP_force_sample_{i}_regular+map.png"))

# （可选）列出目录中文件，确认已写入
print("\n目录文件：")
for fn in os.listdir(shap_dir):
    print(" *", fn)


# In[45]:


# ====================== SVM (Pipeline + AUC选参 + Fβ阈值 + CV汇总) ======================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, brier_score_loss, fbeta_score
)
from sklearn.calibration import calibration_curve

# ---------- 你的数据 ----------
# X_train, X_test: pandas.DataFrame
# y_train, y_test: pandas.Series（index与X一致）

RANDOM_STATE   = 42
TARGET_RECALL  = None    # 例如 0.70；若无硬约束设为 None
CV_SPLITS      = 5
root_dir       = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii"
roc_dir        = os.path.join(root_dir, "ROC_regular")
save_dir_svm   = os.path.join(root_dir, "SVM")
os.makedirs(roc_dir, exist_ok=True)
os.makedirs(save_dir_svm, exist_ok=True)

# ====================== 工具函数 ======================
def find_best_youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    J = tpr - fpr
    best = np.where(J == J.max())[0]
    return float(np.median(thresholds[best])), float(J.max())
    
def cv_threshold_by_metric(X, y, best_params, n_splits=5, seed=RANDOM_STATE):
    """K折：每折只用该折训练子集拟合同结构Pipeline，并在该子集上用Youden策略定阈值；最后取“中位数”作为固定阈值。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ths = []
    for tr_idx, _ in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        # 用最优参数构建同结构Pipeline
        svc = SVC(probability=True, class_weight='balanced', random_state=seed)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", svc)
        ])
        pipe.set_params(**best_params)  # best_params 来自 GridSearchCV.best_params_
        pipe.fit(X_tr, y_tr)
        p_tr = pipe.predict_proba(X_tr)[:, 1]
        th, _ = find_best_youden_threshold(y_tr, p_tr)
        ths.append(th)
    return float(np.median(ths))

def evaluate_binary_classifier(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_true, y_prob)
    return {"Accuracy":acc,"Sensitivity":sens,"Specificity":spec,"F1":f1,
            "Recall":rec,"Precision":prec,"AUC":auc,"Brier":brier,"Threshold":threshold}

def print_metrics_classic(train_metrics: dict, test_metrics: dict,
                          best_params: dict = None, final_threshold: float = None,
                          prefix: str = "SVM"):
    if best_params is not None:
        print(f"Best params from cross-validation: {best_params}")
    if final_threshold is not None:
        print(f"Fixed threshold (Youden via CV median): {final_threshold:.4f}")

    print(f"{prefix} Train Accuracy: {train_metrics.get('Accuracy', float('nan')):.4f}")
    print(f"{prefix} Test Accuracy: {test_metrics.get('Accuracy', float('nan')):.4f}")

    print(f"{prefix} Train Sensitivity (Recall): {train_metrics.get('Sensitivity', float('nan')):.4f}")
    print(f"{prefix} Test Sensitivity (Recall): {test_metrics.get('Sensitivity', float('nan')):.4f}")

    print(f"{prefix} Train Specificity: {train_metrics.get('Specificity', float('nan')):.4f}")
    print(f"{prefix} Test Specificity: {test_metrics.get('Specificity', float('nan')):.4f}")

    print(f"{prefix} Train AUC: {train_metrics.get('AUC', float('nan')):.3f}")
    print(f"{prefix} Test  AUC: {test_metrics.get('AUC', float('nan')):.3f}")

    print(f"{prefix} Train F1 Score:      {train_metrics.get('F1', float('nan')):.4f}")
    print(f"{prefix} Test  F1 Score:      {test_metrics.get('F1', float('nan')):.4f}")

    print(f"{prefix} Train Recall:        {train_metrics.get('Recall', float('nan')):.4f}")
    print(f"{prefix} Test  Recall:        {test_metrics.get('Recall', float('nan')):.4f}")

    print(f"{prefix} Train Precision:     {train_metrics.get('Precision', float('nan')):.4f}")
    print(f"{prefix} Test  Precision:     {test_metrics.get('Precision', float('nan')):.4f}")

    print(f"{prefix} Train Brier Score:   {train_metrics.get('Brier', float('nan')):.4f}")
    print(f"{prefix} Test  Brier Score:   {test_metrics.get('Brier', float('nan')):.4f}")

def save_radiomics_score(path, index, scores, labels=None, verbose=True):
    df = pd.DataFrame({'PatientID': index, 'RadiomicsScore': scores})
    if labels is not None:
        labels_aligned = pd.Series(labels).reindex(index)
        df["Label"] = labels_aligned.values
    df.to_excel(path, index=False)
    if verbose:
        print(f"Saved: {path}")
    return df

def plot_roc(y_true, y_prob, title, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def bootstrap_metrics_fixed_model(model, X, y, threshold, n_iter=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    probs = model.predict_proba(X)[:, 1]
    out = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y), len(y))
        y_b = y.iloc[idx]
        p_b = probs[idx]
        yhat = (p_b >= threshold).astype(int)
        if y_b.nunique() < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_b, yhat, labels=[0,1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        f1  = f1_score(y_b, yhat, zero_division=0)
        auc = roc_auc_score(y_b, p_b)
        out.append([auc, acc, rec, spec, f1])
    arr = np.array(out)
    means = arr.mean(axis=0)
    lows, highs = np.percentile(arr, [2.5, 97.5], axis=0)
    return means, np.vstack([lows, highs]).T

def print_bootstrap_percentile(name, means, ci):
    for metric_name, mean, (low, high) in zip(["AUC","Acc","Recall","Spec","F1"], means, ci):
        print(f"{metric_name}: {mean:.3f} (95% CI: {low:.3f} ~ {high:.3f})")

# ====================== 模型选择（Pipeline + GridSearchCV, AUC评分） ======================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))
])


param_grid = [
    {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]}
]
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="roc_auc", refit=True)
gs.fit(X_train, y_train)

best_params = gs.best_params_
best_model  = gs.best_estimator_

# ====================== 概率 & 阈值（Fβ + CV汇总） ======================
y_train_prob = best_model.predict_proba(X_train)[:, 1]
y_test_prob  = best_model.predict_proba(X_test)[:, 1]

final_threshold = cv_threshold_by_metric(
    X_train, y_train, best_params,
    n_splits=CV_SPLITS, seed=RANDOM_STATE
)

# ====================== 固定阈值评估 ======================
train_metrics = evaluate_binary_classifier(y_train, y_train_prob, final_threshold)
test_metrics  = evaluate_binary_classifier(y_test,  y_test_prob,  final_threshold)

print_metrics_classic(train_metrics, test_metrics,
                      best_params=best_params, final_threshold=final_threshold, prefix="SVM")

# ====================== Radiomics Score 保存 ======================
train_radiomics_df = save_radiomics_score(
    os.path.join(save_dir_svm, "SVM_train_radiomics_scores_regular+map.xlsx"),
    index=X_train.index, scores=y_train_prob, labels=y_train
)
test_radiomics_df = save_radiomics_score(
    os.path.join(save_dir_svm, "SVM_test_radiomics_scores_regular+map.xlsx"),
    index=X_test.index,  scores=y_test_prob,  labels=y_test
)

print("\nTrain Radiomics Scores (head):")
print(train_radiomics_df.head())
print("\nTest Radiomics Scores (head):")
print(test_radiomics_df.head())
print("\nRadiomics Score 已保存！")

# ====================== ROC ======================
plot_roc(y_train, y_train_prob, "SVM Train ROC", os.path.join(roc_dir, "SVM_train_ROC.png"))
plot_roc(y_test,  y_test_prob,  "SVM Test ROC",  os.path.join(roc_dir, "SVM_test_ROC.png"))

# ====================== Bootstrap 百分位CI（固定模型+样本扰动） ======================
means_tr, ci_tr = bootstrap_metrics_fixed_model(best_model, X_train, y_train, final_threshold, n_iter=500)
means_te, ci_te = bootstrap_metrics_fixed_model(best_model, X_test,  y_test,  final_threshold, n_iter=500)

print("\nBootstrap summary (Train, percentile CI):")
print_bootstrap_percentile("Train", means_tr, ci_tr)
print("\nBootstrap summary (Test, percentile CI):")
print_bootstrap_percentile("Test", means_te, ci_te)


# In[47]:


# XGB (focal-loss Booster) — 统一版训练/评估/可视化/Bootstrap
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import partial
from itertools import product
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, roc_curve,
    f1_score, recall_score, precision_score
)
from sklearn.calibration import calibration_curve

# ========= 工具函数 =========
def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity = recall
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec

def find_best_f1_threshold(y_true, y_prob, n_grid=200):
    thresholds = np.linspace(0.0, 1.0, n_grid, endpoint=False)[1:]  # 避免0阈值
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])

def save_radiomics_score(path, index, scores, labels=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({'PatientID': index, 'RadiomicsScore': scores})
    if labels is not None:
        labels = pd.Series(labels).reindex(index)  # 严格对齐
        df["Label"] = labels.values
    df.to_excel(path, index=False)
    print(f"[Radiomics Score Saved] {path}")

# focal loss（logit输入）
def focal_loss_obj(predt, dtrain, alpha=0.5, gamma=2.0):
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pt = y * p + (1 - y) * (1 - p)
    w = alpha * y + (1 - alpha) * (1 - y)

    dFL_dp_pos = -alpha * ( -gamma * (1 - p)**(gamma - 1) * np.log(p) + (1 - p)**gamma / p )
    dFL_dp_neg = -(1 - alpha) * (  gamma * p**(gamma - 1) * np.log(1 - p) - p**gamma / (1 - p) )
    dFL_dp = y * dFL_dp_pos + (1 - y) * dFL_dp_neg
    grad = dFL_dp * p * (1 - p)  # d p / d predt

    hess = w * ((1 - pt) ** gamma) * p * (1 - p) * (1 + gamma * (1 - pt))
    hess = np.clip(hess, 1e-12, None)
    return grad, hess

def print_metrics_table(prefix, y_true_tr, y_pred_tr, y_prob_tr,
                        y_true_te, y_pred_te, y_prob_te):
    sens_tr, spec_tr = sens_spec(y_true_tr, y_pred_tr)
    sens_te, spec_te = sens_spec(y_true_te, y_pred_te)

    print(f"\n{prefix} Train Accuracy:             {accuracy_score(y_true_tr, y_pred_tr):.4f}")
    print(f"{prefix} Test  Accuracy:             {accuracy_score(y_true_te, y_pred_te):.4f}")
    print(f"{prefix} Train Sensitivity (Recall): {sens_tr:.4f}")
    print(f"{prefix} Test  Sensitivity (Recall): {sens_te:.4f}")
    print(f"{prefix} Train Specificity:          {spec_tr:.4f}")
    print(f"{prefix} Test  Specificity:          {spec_te:.4f}")
    print(f"{prefix} Train AUC:                  {roc_auc_score(y_true_tr, y_prob_tr):.3f}")
    print(f"{prefix} Test  AUC:                  {roc_auc_score(y_true_te, y_prob_te):.3f}")
    print(f"{prefix} Train F1 Score:             {f1_score(y_true_tr, y_pred_tr, zero_division=0):.4f}")
    print(f"{prefix} Test  F1 Score:             {f1_score(y_true_te, y_pred_te, zero_division=0):.4f}")
    print(f"{prefix} Train Precision:            {precision_score(y_true_tr, y_pred_tr, zero_division=0):.4f}")
    print(f"{prefix} Test  Precision:            {precision_score(y_true_te, y_pred_te, zero_division=0):.4f}")

# 固定模型 + 样本扰动 的自助法
def bootstrap_test_uncertainty_fixed_model(model, X, y, threshold, n_iter=200, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    # 取概率（兼容 Booster / sklearn）
    if isinstance(model, xgb.Booster):
        y_prob_all = model.predict(xgb.DMatrix(X))
    else:
        y_prob_all = model.predict_proba(X)[:, 1]

    metrics = []
    for _ in range(n_iter):
        idx = rng.choice(len(y), size=len(y), replace=True)
        y_true_boot = y[idx]
        y_prob_boot = y_prob_all[idx]
        y_pred_boot = (y_prob_boot >= threshold).astype(int)

        # 保证两类都有
        if len(np.unique(y_true_boot)) < 2:
            continue

        auc = roc_auc_score(y_true_boot, y_prob_boot)
        acc = accuracy_score(y_true_boot, y_pred_boot)
        rec = recall_score(y_true_boot, y_pred_boot, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = f1_score(y_true_boot, y_pred_boot, zero_division=0)
        metrics.append([auc, acc, rec, spec, f1])

    return np.array(metrics)

def print_bootstrap_results(name, metrics):
    if metrics.size == 0:
        print(f"\nBootstrapping {name}: 样本过小或类别极不平衡，未形成有效自助样本。")
        return
    print(f"\nBootstrapping {name}:")
    for i, metric_name in enumerate(["AUC", "Acc", "Recall", "Spec", "F1"]):
        vals = metrics[:, i]
        mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        lower, upper = mean - 1.96*std, mean + 1.96*std
        print(f"{metric_name}: {mean:.3f} (95% CI: {lower:.3f} ~ {upper:.3f})")

# ========= 数据准备（假定你已有 X_train, y_train, X_test, y_test）=========
# 不做SMOTE，树模型无需标准化
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# 类别不平衡权重（负/正）
# pos = int((np.asarray(y_train) == 1).sum())
# neg = int((np.asarray(y_train) == 0).sum())
# scale_pos_weight = neg / max(1, pos)
# print("Scale Pos Weight:", scale_pos_weight)

alpha, gamma = 0.75, 2.0   # 正类少数 → 给正类更高权重
print(f"Using focal loss with alpha={alpha}, gamma={gamma}")

# ========= 超参搜索（与最终模型同一：focal-loss + eval_metric=auc）=========
base_params = {
    'objective': 'binary:logistic',  # 预测概率
    'eval_metric': ['auc','aucpr'],
    'tree_method': 'hist',           # 视硬件可改为 'gpu_hist'
    #'scale_pos_weight': scale_pos_weight,
    'verbosity': 0,
    'seed': 42
}

search_space = {
    'eta':              [0.03, 0.05, 0.1],
    'max_depth':        [3, 4, 5],
    'min_child_weight': [1, 3],
    'subsample':        [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'reg_alpha':        [0.0, 0.01, 0.1],
    'reg_lambda':       [1.0, 0.1]
}

best_score = -np.inf
best_params = None
best_rounds = None

combo_list = list(product(*search_space.values()))
print(f"CV candidates: {len(combo_list)}")
for i, values in enumerate(combo_list, 1):
    params = base_params.copy()
    for k, v in zip(search_space.keys(), values):
        params[k] = v

    # 用 xgb.cv 与 focal loss；早停自动确定最佳轮数
    cvres = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=42
    )
    score = cvres['test-auc-mean'].iloc[-1]
    rounds = cvres.shape[0]

    if score > best_score:
        best_score = score
        best_params = params.copy()
        best_rounds = rounds

    if i % 10 == 0 or i == len(combo_list):
        print(f"  Progress: {i}/{len(combo_list)} | current best AUC={best_score:.4f} @ rounds={best_rounds}")

print("\nBest Params (CV on focal-loss):")
pretty = {k: best_params[k] for k in search_space.keys()}
print(pretty)
print(f"Best CV AUC: {best_score:.4f}  | Best rounds: {best_rounds}")

# ========= 最终训练（使用同一套 focal-loss + 最佳轮数）=========
bst = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=best_rounds,
    obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
    verbose_eval=False
)

# ========= 评估与可视化 =========
# 概率
y_train_prob = bst.predict(dtrain)
y_test_prob  = bst.predict(dtest)

# 固化阈值：训练集最大F1
thresh, best_f1 = find_best_f1_threshold(y_train, y_train_prob)
print(f"\n固化最终分类阈值为训练集最大 F1: {thresh:.3f} (F1 = {best_f1:.4f})")

# 标签
y_train_pred = (y_train_prob >= thresh).astype(int)
y_test_pred  = (y_test_prob  >= thresh).astype(int)

# 指标
print_metrics_table("XGB", y_train, y_train_pred, y_train_prob,
                    y_test,  y_test_pred,  y_test_prob)

# ROC（Train & Test）
fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_prob)
fpr_te, tpr_te, _ = roc_curve(y_test,  y_test_prob)
auc_tr = roc_auc_score(y_train, y_train_prob)
auc_te = roc_auc_score(y_test,  y_test_prob)

plt.figure(figsize=(7,7))
plt.plot(fpr_tr, tpr_tr, lw=2, label=f"Train ROC (AUC={auc_tr:.3f})")
plt.plot(fpr_te, tpr_te, lw=2, label=f"Test  ROC (AUC={auc_te:.3f})")
plt.plot([0,1],[0,1],'--',c='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Train & Test)")
plt.legend(loc='lower right')
plt.grid(True); plt.tight_layout(); plt.show()

# 校准曲线（直接使用 bst 概率；与主模型一致）
true_tr, pred_tr = calibration_curve(y_train, y_train_prob, n_bins=10, strategy='quantile')
true_te, pred_te = calibration_curve(y_test,  y_test_prob,  n_bins=10, strategy='quantile')

plt.figure(figsize=(6,6))
plt.plot(pred_tr, true_tr, 'o-', label='Train (bst)')
plt.plot([0,1],[0,1],'--',c='gray', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability'); plt.ylabel('True Probability')
plt.title('Calibration Curve (Train)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,6))
plt.plot(pred_te, true_te, 's-', label='Test (bst)')
plt.plot([0,1],[0,1],'--',c='gray', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability'); plt.ylabel('True Probability')
plt.title('Calibration Curve (Test)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ========= Radiomics 分数保存 =========
save_dir = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\XGB"
save_radiomics_score(os.path.join(save_dir, "XGB_train_radiomics_scores_regular+map.xlsx"), X_train.index, y_train_prob, y_train)
save_radiomics_score(os.path.join(save_dir, "XGB_test_radiomics_scores_regular+map.xlsx"),  X_test.index,  y_test_prob,  y_test)

# ========= Bootstrapping（固定模型 + 样本扰动）=========
train_boot = bootstrap_test_uncertainty_fixed_model(bst, X_train, y_train, thresh, n_iter=200, random_state=42)
test_boot  = bootstrap_test_uncertainty_fixed_model(bst, X_test,  y_test,  thresh, n_iter=200, random_state=42)

print_bootstrap_results("训练集", train_boot)
print_bootstrap_results("测试集", test_boot)


# In[27]:


import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 前提：你已经准备好 X_train, y_train, X_test, y_test
RANDOM_STATE = 42
CV_SPLITS = 5
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ----------------------------
# 1) LR（沿用你原来的 GridSearchCV 流程）
# ----------------------------
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=10000))
])

lr_param_grid = [
    {
        "lr__solver": ["liblinear"],
        "lr__penalty": ["l2"],
        "lr__C": np.logspace(-2, 0, 100),
        "lr__class_weight": ["balanced", None]
    },
]

lr_gs = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
lr_gs.fit(X_train, y_train)
lr_cv_auc = float(lr_gs.best_score_)          # 最高 mean CV-AUC
lr_best_model = lr_gs.best_estimator_
lr_best_params = lr_gs.best_params_

# ----------------------------
# 2) RF（沿用你原来的 GridSearchCV 流程）
# ----------------------------
rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 8, 12],
    "min_samples_split": [8, 12],
    "max_features": ["sqrt", 0.3],
    "max_samples": [0.6, 0.7, 0.8],
    "bootstrap": [True],
    "class_weight": ["balanced"]
}
rf_gs = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
rf_gs.fit(X_train, y_train)
rf_cv_auc = float(rf_gs.best_score_)          # 最高 mean CV-AUC
rf_best_params = rf_gs.best_params_
rf_best_model = RandomForestClassifier(**rf_best_params, random_state=RANDOM_STATE).fit(X_train, y_train)

# ----------------------------
# 3) SVM（沿用你原来的 GridSearchCV 流程）
# ----------------------------
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
])
svm_param_grid = [
    {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]}
]
svm_gs = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
svm_gs.fit(X_train, y_train)
svm_cv_auc = float(svm_gs.best_score_)        # 最高 mean CV-AUC
svm_best_model = svm_gs.best_estimator_
svm_best_params = svm_gs.best_params_

print("\n=== Partial results (LR/RF/SVM) ===")
print(f"LR   mean CV-AUC = {lr_cv_auc:.4f}")
print(f"RF   mean CV-AUC = {rf_cv_auc:.4f}")
print(f"SVM  mean CV-AUC = {svm_cv_auc:.4f}")
# ===== Part 2/2: 训练集5折CV训练/调参 XGBoost，并与 LR/RF/SVM 一起比较选最优 =====

import xgboost as xgb
from functools import partial
from itertools import product

# ----------------------------
# 4) XGBoost（沿用你原来的 xgb.cv + focal loss + early stopping 搜参流程）
# ----------------------------
def focal_loss_obj(predt, dtrain, alpha=0.75, gamma=2.0):
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pt = y * p + (1 - y) * (1 - p)
    w = alpha * y + (1 - alpha) * (1 - y)

    dFL_dp_pos = -alpha * (-gamma * (1 - p) ** (gamma - 1) * np.log(p) + (1 - p) ** gamma / p)
    dFL_dp_neg = -(1 - alpha) * (gamma * p ** (gamma - 1) * np.log(1 - p) - p ** gamma / (1 - p))
    dFL_dp = y * dFL_dp_pos + (1 - y) * dFL_dp_neg
    grad = dFL_dp * p * (1 - p)

    hess = w * ((1 - pt) ** gamma) * p * (1 - p) * (1 + gamma * (1 - pt))
    hess = np.clip(hess, 1e-12, None)
    return grad, hess

dtrain = xgb.DMatrix(X_train, label=y_train)

base_params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr"],
    "tree_method": "hist",
    "verbosity": 0,
    "seed": RANDOM_STATE
}

search_space = {
    "eta":              [0.03, 0.05, 0.1],
    "max_depth":        [3, 4, 5],
    "min_child_weight": [1, 3],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha":        [0.0, 0.01, 0.1],
    "reg_lambda":       [1.0, 0.1]
}

best_score = -np.inf
xgb_best_params = None
xgb_best_rounds = None

combo_list = list(product(*search_space.values()))
for values in combo_list:
    params = base_params.copy()
    for k, v in zip(search_space.keys(), values):
        params[k] = v

    cvres = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        nfold=CV_SPLITS,
        stratified=True,
        obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=RANDOM_STATE
    )

    score = float(cvres["test-auc-mean"].iloc[-1])  # mean CV-AUC
    rounds = int(cvres.shape[0])
    if score > best_score:
        best_score = score
        xgb_best_params = params.copy()
        xgb_best_rounds = rounds

xgb_cv_auc = float(best_score)

# 训练最终 XGBoost 模型（用于后续你自己的评估流程）
xgb_best_model = xgb.train(
    params=xgb_best_params,
    dtrain=dtrain,
    num_boost_round=xgb_best_rounds,
    obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
    verbose_eval=False
)
xgb_params_compact = {k: xgb_best_params.get(k) for k in search_space.keys()}

print("\n=== XGBoost result ===")
print(f"XGBoost mean CV-AUC = {xgb_cv_auc:.4f}")

# ----------------------------
# 5) 统一比较并选择最优分类器（highest mean CV-AUC）
# ----------------------------
cv_auc_dict = {
    "LR": lr_cv_auc,
    "RF": rf_cv_auc,
    "SVM": svm_cv_auc,
    "XGBoost": xgb_cv_auc
}

print("\n=== 5-fold mean cross-validated AUC (training-only) ===")
for name, aucv in sorted(cv_auc_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<8} mean CV-AUC = {aucv:.4f}")

best_clf_name = max(cv_auc_dict, key=cv_auc_dict.get)
best_cv_auc = cv_auc_dict[best_clf_name]
print(f"\nSelected classifier: {best_clf_name} (highest mean CV-AUC = {best_cv_auc:.4f})")

# 输出最终 best_model / best_params（方便你后续统一评估/保存）
if best_clf_name == "LR":
    best_model = lr_best_model
    best_params = lr_best_params
elif best_clf_name == "RF":
    best_model = rf_best_model
    best_params = rf_best_params
elif best_clf_name == "SVM":
    best_model = svm_best_model
    best_params = svm_best_params
else:
    best_model = xgb_best_model
    best_params = xgb_params_compact

print("\nBest params of selected classifier:")
print(best_params)

# 统一概率预测接口（sklearn / xgb.Booster 都可用）
def predict_proba_pos(model, X):
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict_proba(X)[:, 1]

# 示例：得到最终选择模型在 train/test 的概率（不参与模型选择，仅供你后续评估用）
y_train_prob = predict_proba_pos(best_model, X_train)
y_test_prob  = predict_proba_pos(best_model, X_test)
print("\nProba ready: y_train_prob / y_test_prob")


# In[5]:


import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 前提：你已经准备好 X_train, y_train, X_test, y_test
RANDOM_STATE = 42
CV_SPLITS = 5
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def mean_ci_from_scores(scores, alpha=0.05):
    """由每折分数计算 mean 和 mean 的 95% CI（t 分布）"""
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    k = len(scores)
    mean = float(scores.mean())
    if k <= 1:
        return mean, mean, mean, float("nan")
    sd = float(scores.std(ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha/2, df=k - 1))
    half = tcrit * sd / np.sqrt(k)
    return mean, mean - half, mean + half, sd

# ----------------------------
# 1) LR（沿用你原来的 GridSearchCV 流程）
# ----------------------------
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=10000))
])

lr_param_grid = [
    {
        "lr__solver": ["liblinear", "saga"],
        "lr__penalty": ["l1", "l2"],
        "lr__C": np.logspace(-2, 0, 100),
        "lr__class_weight": ["balanced", None]
    },
    {
        "lr__solver": ["lbfgs"],
        "lr__penalty": ["l2"],
        "lr__C": np.logspace(-2, 0, 100),
        "lr__class_weight": ["balanced", None]
    }
]

lr_gs = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
lr_gs.fit(X_train, y_train)
lr_best_model = lr_gs.best_estimator_
lr_best_params = lr_gs.best_params_

lr_fold_scores = cross_val_score(lr_best_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
lr_mean, lr_ci_low, lr_ci_high, lr_sd = mean_ci_from_scores(lr_fold_scores)
lr_cv_auc = lr_mean

# ----------------------------
# 2) RF（沿用你原来的 GridSearchCV 流程）
# ----------------------------
rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 8, 12],
    "min_samples_split": [8, 12],
    "max_features": ["sqrt", 0.3],
    "max_samples": [0.6, 0.7, 0.8],
    "bootstrap": [True],
    "class_weight": ["balanced"]
}
rf_gs = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
rf_gs.fit(X_train, y_train)
rf_best_estimator = rf_gs.best_estimator_
rf_best_params = rf_gs.best_params_

rf_fold_scores = cross_val_score(rf_best_estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_mean, rf_ci_low, rf_ci_high, rf_sd = mean_ci_from_scores(rf_fold_scores)
rf_cv_auc = rf_mean

# 如你后续需要“全训练集拟合后的最终RF模型”
rf_best_model = RandomForestClassifier(**rf_best_params, random_state=RANDOM_STATE).fit(X_train, y_train)

# ----------------------------
# 3) SVM（沿用你原来的 GridSearchCV 流程）
# ----------------------------
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
])
svm_param_grid = [
    {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]}
]
svm_gs = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
svm_gs.fit(X_train, y_train)
svm_best_model = svm_gs.best_estimator_
svm_best_params = svm_gs.best_params_

svm_fold_scores = cross_val_score(svm_best_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
svm_mean, svm_ci_low, svm_ci_high, svm_sd = mean_ci_from_scores(svm_fold_scores)
svm_cv_auc = svm_mean

print("\n=== Partial results (LR/RF/SVM): mean CV-AUC with 95% CI ===")
print(f"LR   mean CV-AUC = {lr_cv_auc:.4f} (95% CI {lr_ci_low:.4f}–{lr_ci_high:.4f}); best_params={lr_best_params}")
print(f"RF   mean CV-AUC = {rf_cv_auc:.4f} (95% CI {rf_ci_low:.4f}–{rf_ci_high:.4f}); best_params={rf_best_params}")
print(f"SVM  mean CV-AUC = {svm_cv_auc:.4f} (95% CI {svm_ci_low:.4f}–{svm_ci_high:.4f}); best_params={svm_best_params}")
# ===== Part 2/2: XGBoost 的 mean CV-AUC + 95% CI + best_params，并与 LR/RF/SVM 一起选最优 =====
import xgboost as xgb
from functools import partial
from itertools import product

# ----------------------------
# 4) XGBoost（沿用你原来的 xgb.cv + focal loss + early stopping 搜参流程）
# ----------------------------
def focal_loss_obj(predt, dtrain, alpha=0.75, gamma=2.0):
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pt = y * p + (1 - y) * (1 - p)
    w = alpha * y + (1 - alpha) * (1 - y)

    dFL_dp_pos = -alpha * (-gamma * (1 - p) ** (gamma - 1) * np.log(p) + (1 - p) ** gamma / p)
    dFL_dp_neg = -(1 - alpha) * (gamma * p ** (gamma - 1) * np.log(1 - p) - p ** gamma / (1 - p))
    dFL_dp = y * dFL_dp_pos + (1 - y) * dFL_dp_neg
    grad = dFL_dp * p * (1 - p)

    hess = w * ((1 - pt) ** gamma) * p * (1 - p) * (1 + gamma * (1 - pt))
    hess = np.clip(hess, 1e-12, None)
    return grad, hess

dtrain = xgb.DMatrix(X_train, label=y_train)

base_params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr"],
    "tree_method": "hist",
    "verbosity": 0,
    "seed": RANDOM_STATE
}

search_space = {
    "eta":              [0.03, 0.05, 0.1],
    "max_depth":        [3, 4, 5],
    "min_child_weight": [1, 3],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha":        [0.0, 0.01, 0.1],
    "reg_lambda":       [1.0, 0.1]
}

best_score = -np.inf
xgb_best_params = None
xgb_best_rounds = None
best_cvres = None

combo_list = list(product(*search_space.values()))
for values in combo_list:
    params = base_params.copy()
    for k, v in zip(search_space.keys(), values):
        params[k] = v

    cvres = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        nfold=CV_SPLITS,
        stratified=True,
        obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=RANDOM_STATE
    )

    score = float(cvres["test-auc-mean"].iloc[-1])  # mean CV-AUC
    rounds = int(cvres.shape[0])

    if score > best_score:
        best_score = score
        xgb_best_params = params.copy()
        xgb_best_rounds = rounds
        best_cvres = cvres

xgb_cv_auc = float(best_score)

# XGB 的 std 来自 best_cvres 的最后一行（跨折标准差）
xgb_std = float(best_cvres["test-auc-std"].iloc[-1])
tcrit = float(stats.t.ppf(1 - 0.05/2, df=CV_SPLITS - 1))
xgb_half = tcrit * xgb_std / np.sqrt(CV_SPLITS)
xgb_ci_low, xgb_ci_high = xgb_cv_auc - xgb_half, xgb_cv_auc + xgb_half

# 训练最终 XGBoost 模型（用于后续评估）
xgb_best_model = xgb.train(
    params=xgb_best_params,
    dtrain=dtrain,
    num_boost_round=xgb_best_rounds,
    obj=partial(focal_loss_obj, alpha=0.75, gamma=2.0),
    verbose_eval=False
)
xgb_params_compact = {k: xgb_best_params.get(k) for k in search_space.keys()}

print("\n=== XGBoost result: mean CV-AUC with 95% CI ===")
print(f"XGBoost mean CV-AUC = {xgb_cv_auc:.4f} (95% CI {xgb_ci_low:.4f}–{xgb_ci_high:.4f}); "
      f"best_rounds={xgb_best_rounds}; best_params={xgb_params_compact}")

# ----------------------------
# 5) 统一比较并选择最优分类器（highest mean CV-AUC）
# ----------------------------
cv_auc_dict = {
    "LR": lr_cv_auc,
    "RF": rf_cv_auc,
    "SVM": svm_cv_auc,
    "XGBoost": xgb_cv_auc
}
cv_ci_dict = {
    "LR": (lr_ci_low, lr_ci_high),
    "RF": (rf_ci_low, rf_ci_high),
    "SVM": (svm_ci_low, svm_ci_high),
    "XGBoost": (xgb_ci_low, xgb_ci_high),
}
best_params_dict = {
    "LR": lr_best_params,
    "RF": rf_best_params,
    "SVM": svm_best_params,
    "XGBoost": {"best_rounds": xgb_best_rounds, **xgb_params_compact},
}

print("\n=== 5-fold mean cross-validated AUC (training-only): with 95% CI & best_params ===")
for name, aucv in sorted(cv_auc_dict.items(), key=lambda x: x[1], reverse=True):
    lo, hi = cv_ci_dict[name]
    print(f"{name:<8} mean CV-AUC = {aucv:.4f} (95% CI {lo:.4f}–{hi:.4f}) | best_params = {best_params_dict[name]}")

best_clf_name = max(cv_auc_dict, key=cv_auc_dict.get)
best_cv_auc = cv_auc_dict[best_clf_name]
best_ci = cv_ci_dict[best_clf_name]
print(f"\nSelected classifier: {best_clf_name} (highest mean CV-AUC = {best_cv_auc:.4f}; "
      f"95% CI {best_ci[0]:.4f}–{best_ci[1]:.4f})")

# 输出最终 best_model / best_params（便于你后续统一评估/保存）
if best_clf_name == "LR":
    best_model = lr_best_model
    best_params = lr_best_params
elif best_clf_name == "RF":
    best_model = rf_best_model
    best_params = rf_best_params
elif best_clf_name == "SVM":
    best_model = svm_best_model
    best_params = svm_best_params
else:
    best_model = xgb_best_model
    best_params = {"best_rounds": xgb_best_rounds, **xgb_params_compact}

print("\nBest params of selected classifier:")
print(best_params)

# 统一概率预测接口（sklearn / xgb.Booster 都可用）
def predict_proba_pos(model, X):
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict_proba(X)[:, 1]

# 示例：得到最终选择模型在 train/test 的概率（不参与模型选择，仅供你后续评估用）
y_train_prob = predict_proba_pos(best_model, X_train)
y_test_prob  = predict_proba_pos(best_model, X_test)
print("\nProba ready: y_train_prob / y_test_prob")


# In[ ]:




