# language: python
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             precision_score, roc_auc_score, brier_score_loss,
                             confusion_matrix)
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

clinical_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\radiograhic_label.xlsx"
train_radiomics_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_train_radiomics_scores_regular+map.xlsx"
test_radiomics_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_test_radiomics_scores_regular+map.xlsx"
fusion_out_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\fusion.xlsx"

clinical = pd.read_excel(clinical_path)
clinical.columns = [str(c).strip() for c in clinical.columns]
assert "PatientID" in clinical.columns and "Label" in clinical.columns, "临床表必须包含 PatientID 与 Label"
clinical['PatientID'] = clinical['PatientID'].astype(str).str.strip()
clinical = clinical.set_index("PatientID")

train_rad = pd.read_excel(train_radiomics_path)
train_rad['PatientID'] = train_rad['PatientID'].astype(str).str.strip()
train_ids = train_rad['PatientID'].tolist()

test_rad = pd.read_excel(test_radiomics_path)
test_rad['PatientID'] = test_rad['PatientID'].astype(str).str.strip()
test_ids = test_rad['PatientID'].tolist()

train_df = clinical.loc[train_ids].copy()
test_df = clinical.loc[test_ids].copy()

y_train = train_df["Label"].astype(int)
y_test = test_df["Label"].astype(int)

drop_cols = {"Label"}
maybe_id_cols = {c for c in train_df.columns if "id" in c.lower()}
X_train_raw = train_df.drop(columns=list(drop_cols | maybe_id_cols), errors="ignore")
X_test_raw = test_df.drop(columns=list(drop_cols | maybe_id_cols), errors="ignore")

num_cols = [c for c in X_train_raw.columns if pd.api.types.is_numeric_dtype(X_train_raw[c])]
obj_cols = [c for c in X_train_raw.columns if X_train_raw[c].dtype == "object"]
num_low_level = [c for c in num_cols if X_train_raw[c].nunique(dropna=True) <= 5]
cat_cols = sorted(list(set(obj_cols) | set(num_low_level)))
cont_cols = sorted([c for c in num_cols if c not in num_low_level])

uni_rows = []
uni_results_list = []

print("\n" + "="*50)
print(" " * 17 + "单因素回归分析")
print("="*50)
print(f"{'Feature':<25} {'OR (95% CI)':<25} {'P-value':<10}")
print("-" * 60)

for c in cont_cols:
    s = X_train_raw[c].dropna()
    y_s = y_train[s.index]
    if len(y_s.unique()) < 2: continue

    model = sm.Logit(y_s, sm.add_constant(s)).fit(disp=0)
    p_value = model.pvalues[c]
    coef = model.params[c]
    conf = model.conf_int()
    or_val = np.exp(coef)
    or_ci = f"{np.exp(conf.loc[c, 0]):.3f} - {np.exp(conf.loc[c, 1]):.3f}"

    uni_rows.append({"feature": c, "p": p_value})
    print(f"{c:<25} {f'{or_val:.3f} ({or_ci})':<25} {p_value:<10.4f}")

for c in cat_cols:
    s = X_train_raw[[c]].dropna()
    y_s = y_train[s.index]
    if len(y_s.unique()) < 2 or s[c].nunique() < 2: continue

    s_dummies = pd.get_dummies(s.astype(str), drop_first=True, prefix=c)
    s_dummies = s_dummies.astype(float)
    y_s_np = y_s.to_numpy()
    for col in s_dummies.columns:
        try:
            model = sm.Logit(y_s_np, sm.add_constant(s_dummies[col])).fit(disp=0)
            p_value = model.pvalues[col]
            coef = model.params[col]
            conf = model.conf_int()
            or_val = np.exp(coef)
            or_ci = f"{np.exp(conf.loc[col, 0]):.3f} - {np.exp(conf.loc[col, 1]):.3f}"

      
            if c not in [row['feature'] for row in uni_rows]:
                 uni_rows.append({"feature": c, "p": p_value})
            print(f"{col:<25} {f'{or_val:.3f} ({or_ci})':<25} {p_value:<10.4f}")
        except Exception as e:
            print(f"{col:<25} Error: {e}")
            # Skip this dummy variable

uni_df = pd.DataFrame(uni_rows).sort_values("p")
candidates = uni_df.loc[uni_df["p"] < 0.10, "feature"].tolist()
print(f"单因素分析后进入多因素的候选变量 (p<0.1): {candidates}")

selected_features = []
if candidates:
    X_multi = X_train_raw[candidates].copy()

    for col in X_multi.columns:
        if pd.api.types.is_numeric_dtype(X_multi[col]):
            X_multi[col] = X_multi[col].fillna(X_multi[col].median())
        else:
            X_multi[col] = X_multi[col].fillna(X_multi[col].mode()[0])

            X_multi[col] = X_multi[col].astype(str)

    X_multi_cont = X_multi.select_dtypes(include=np.number)
    X_multi_cat = X_multi.select_dtypes(exclude=np.number)


    if not X_multi_cat.empty:
        X_multi_cat_dummies = pd.get_dummies(X_multi_cat, drop_first=True, dummy_na=False)
    
        X_for_sm_unconst = pd.concat([X_multi_cont, X_multi_cat_dummies], axis=1)
    else:
        X_for_sm_unconst = X_multi_cont

    X_for_sm_unconst = X_for_sm_unconst.astype(float)
    X_for_sm = sm.add_constant(X_for_sm_unconst, has_constant='add')

    def backward_selection(X, y):
        included = list(X.columns)
        if 'const' in included: included.remove('const')
        while True:
            changed = False
            model = sm.Logit(y, X[['const'] + included]).fit(disp=False)
            current_aic = model.aic
            aics = []
            for cand in included:
                reduced = [v for v in included if v != cand]
                try:
                    m = sm.Logit(y, X[['const'] + reduced]).fit(disp=False)
                    aics.append((cand, m.aic))
                except Exception: continue
            if not aics: break
            aics.sort(key=lambda x: x[1])
            best_cand, best_aic = aics[0]
            if best_aic < current_aic:
                included.remove(best_cand)
                changed = True
            if not changed: break
        return included

    try:
        final_vars_encoded = backward_selection(X_for_sm, y_train)

        final_model = sm.Logit(y_train, X_for_sm[['const'] + final_vars_encoded]).fit(disp=0)

        print("\n" + "="*50)
        print(" " * 17 + "多因素回归分析")
        print("="*50)
        print(f"{'Feature':<25} {'OR (95% CI)':<25} {'P-value':<10}")
        print("-" * 60)

        params = final_model.params
        conf = final_model.conf_int()
        pvalues = final_model.pvalues

        for var in params.index:
            if var == 'const': continue
            or_val = np.exp(params[var])
            or_ci = f"{np.exp(conf.loc[var, 0]):.3f} - {np.exp(conf.loc[var, 1]):.3f}"
            print(f"{var:<25} {f'{or_val:.3f} ({or_ci})':<25} {pvalues[var]:<10.4f}")

  
        original_features = set()
        for var in final_vars_encoded:
 
            if '_' in var and any(var.startswith(cat + '_') for cat in cat_cols):
                original_features.add(var.split('_')[0])
            else:
                original_features.add(var)
        selected_features = list(original_features)

    except Exception as e:
        print(f"多因素分析失败: {e}")
        selected_features = []

print(f"多因素分析后最终选择的临床特征: {selected_features}")



score_col = [c for c in train_rad.columns if "radiomics" in c.lower() or "score" in c.lower()][0]


X_train_final = pd.DataFrame(index=train_ids)
if selected_features:
    X_train_final = X_train_raw.loc[train_ids, [c for c in X_train_raw.columns if c in selected_features]].copy()

X_train_final[score_col] = train_rad.set_index('PatientID').loc[train_ids, score_col]


X_test_final = pd.DataFrame(index=test_ids)
if selected_features:
    X_test_final = X_test_raw.loc[test_ids, [c for c in X_test_raw.columns if c in selected_features]].copy()
X_test_final[score_col] = test_rad.set_index('PatientID').loc[test_ids, score_col]


model_cols_cat = [c for c in X_train_final.columns if X_train_final[c].dtype == "object" or X_train_final[c].nunique() <=5]
model_cols_num = [c for c in X_train_final.columns if c not in model_cols_cat]

transformers = []
if model_cols_num:
    transformers.append(("num", StandardScaler(), model_cols_num))
if model_cols_cat:
    transformers.append(("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), model_cols_cat))

if not transformers:
  
    X_train_final = X_train_final[[score_col]]
    X_test_final = X_test_final[[score_col]]
    pre = StandardScaler()
else:
    pre = ColumnTransformer(transformers, remainder='passthrough')


clf = Pipeline([
    ("pre", pre),
    ("logreg", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced'))
])


clf.fit(X_train_final, y_train)

train_proba = clf.predict_proba(X_train_final)[:, 1]
test_proba = clf.predict_proba(X_test_final)[:, 1]

print("\n" + "="*50)
print(" " * 15 + "模型性能评估")
print("="*50)

y_train_pred = (train_proba >= 0.5).astype(int)
y_test_pred = (test_proba >= 0.5).astype(int)

def calculate_and_print_metrics(y_true, y_pred, y_proba, dataset_name):
    """计算并打印所有指定的性能指标"""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
    except ValueError: 
        specificity = 0

    print(f"LR {dataset_name} Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"LR {dataset_name} Sensitivity (Recall): {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"LR {dataset_name} Specificity: {specificity:.4f}")
    print(f"LR {dataset_name} AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"LR {dataset_name} F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"LR {dataset_name} Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"LR {dataset_name} Brier Score: {brier_score_loss(y_true, y_proba):.4f}")
    print("-" * 25)

calculate_and_print_metrics(y_train, y_train_pred, train_proba, "Train")

calculate_and_print_metrics(y_test, y_test_pred, test_proba, "Test")

print("\n" + "="*50)
print(" " * 12 + "Bootstrapping (1000次)")
print("="*50)

def bootstrap_metrics(y_true, y_proba, n_bootstraps=1000):
    """执行 bootstrapping 并计算置信区间"""
    n_samples = len(y_true)
    metrics = {'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'Precision': []}

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)

        if len(np.unique(y_true[indices])) < 2:
            continue 

        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]
        y_pred_boot = (y_proba_boot >= 0.5).astype(int)

        metrics['AUC'].append(roc_auc_score(y_true_boot, y_proba_boot))
        metrics['Accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics['Sensitivity'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))

        try:
            tn, fp, _, _ = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['Specificity'].append(specificity)
        except ValueError:
            metrics['Specificity'].append(0)

        metrics['F1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['Precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))

    print(f"{'Metric':<12} {'Mean':<10} {'95% CI':<20}")
    print("-" * 45)
    for key, values in metrics.items():
        if values:
            mean_val = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            print(f"{key:<12} {mean_val:<10.3f} {f'({ci_lower:.3f} - {ci_upper:.3f})':<20}")

print("--- Train Set ---")
bootstrap_metrics(y_train.values, train_proba)

print("\n--- Test Set ---")
bootstrap_metrics(y_test.values, test_proba)

train_res = pd.DataFrame({
    'PatientID': train_ids,
    'RadiomicsScore': train_proba,
    'Label': y_train.values
})

test_res = pd.DataFrame({
    'PatientID': test_ids,
    'RadiomicsScore': test_proba,
    'Label': y_test.values
})

train_fusion_out_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_train_radiomics_scores_fusion.xlsx"
test_fusion_out_path = r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\LR\LR_test_radiomics_scores_fusion.xlsx"


train_res = train_res[['PatientID', 'RadiomicsScore', 'Label']]
test_res = test_res[['PatientID', 'RadiomicsScore', 'Label']]


train_res.to_excel(train_fusion_out_path, index=False)
test_res.to_excel(test_fusion_out_path, index=False)

print(f"\n模型训练和预测完成。")
print(f"训练集融合分数已保存到: {train_fusion_out_path}")
print(f"测试集融合分数已保存到: {test_fusion_out_path}")
