import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import os

# 文件路径
ro_all_merged_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ro_all_merged_regular+map.xlsx"  # 包含所有特征和病人数据
lasso_selected_features_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\lasso_selected_features_regular+map.txt"  # Lasso选择的特征
# rfe_selected_features_file = r"C:\Users\Sun\Desktop\3dslicer_tougao_malignant_nii\rfe_selected_features.txt"
train_set_txt_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\train_regular+map.txt"  # 训练集住院号
test_set_txt_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\test_regular+map.txt"  # 测试集住院号
patient_label_file = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\patient_label.xlsx"  # 标签文件


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
        plt.savefig(os.path.join(r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_regular+map",
                                 feature_set + "_" + model_name))
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

# ------------------------------------------------------LR-------------------------------------------------------------
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


# def print_metrics(train_metrics, test_metrics, best_C):
#     print(f"\nBest C value from cross-validation: {best_C:.6f}")
#     for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
#         print(f"LR {name} Accuracy: {m['Accuracy']:.4f}")
#         print(f"LR {name} Sensitivity (Recall): {m['Sensitivity']:.4f}")
#         print(f"LR {name} Specificity: {m['Specificity']:.4f}")
#         print(f"LR {name} AUC: {m['AUC']:.3f}")
#         print(f"LR {name} F1 Score: {m['F1']:.4f}")
#         print(f"LR {name} Precision: {m['Precision']:.4f}")
#         print(f"LR {name} Brier Score: {m['Brier']:.4f}")
#     print()

def print_metrics_classic(train_metrics: dict, test_metrics: dict, prefix: str = "LR"):
    # 为了和你给的样式保持一致：AUC保留3位小数，其余保留4位小数
    print(f"Best C value from cross-validation: {best_C:.6f}")
    print(f"Fixed threshold (Youden on Train): {final_threshold:.4f}")

    print(f"{prefix} Train Accuracy: {train_metrics.get('Accuracy', float('nan')):.4f}")
    print(f"{prefix} Test Accuracy: {test_metrics.get('Accuracy', float('nan')):.4f}")

    print(f"{prefix} Train Sensitivity (Recall): {train_metrics.get('Sensitivity', float('nan')):.4f}")
    print(f"{prefix} Test Sensitivity (Recall): {test_metrics.get('Sensitivity', float('nan')):.4f}")

    print(f"{prefix} Train Specificity: {train_metrics.get('Specificity', float('nan')):.4f}")
    print(f"{prefix} Test Specificity: {test_metrics.get('Specificity', float('nan')):.4f}")

    # 注意你示例里“LR Test  AUC”有两个空格，我也保留这个排版
    print(f"{prefix} Train AUC: {train_metrics.get('AUC', float('nan')):.3f}")
    print(f"{prefix} Test  AUC: {test_metrics.get('AUC', float('nan')):.3f}")

    # 同样保留你示例里 F1 行的空格对齐
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


# Bootstrap with percentile CI
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


def cv_youden_threshold(X, y, best_C, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ths = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        # 用和主模型一致的管线与超参
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=10000,
                class_weight="balanced",
                C=best_C
            ))
        ])
        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]
        th, _ = find_best_youden_threshold(y_tr, p_tr)  # 只用该折训练子集定阈值
        ths.append(th)
    return float(np.median(ths))


# ====================== 主流程 ======================

roc_dir = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_regular+map"
os.makedirs(roc_dir, exist_ok=True)

# Pipeline + GridSearchCV (AUC 打分，避免泄露)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        class_weight="balanced"
    ))
])

param_grid = {"lr__C": np.logspace(-2, 0, 100)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_C = grid.best_params_["lr__C"]

# 概率预测
y_train_prob = best_model.predict_proba(X_train)[:, 1]
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# 固化阈值（训练集 Youden Index）
# final_threshold, youden_val = find_best_youden_threshold(y_train, y_train_prob)
# 替换你原来的阈值行：
final_threshold = cv_youden_threshold(X_train, y_train, best_C)

# 评估
train_metrics = evaluate_binary_classifier(y_train, y_train_prob, final_threshold)
test_metrics = evaluate_binary_classifier(y_test, y_test_prob, final_threshold)
print_metrics_classic(train_metrics, test_metrics, prefix="LR")

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







