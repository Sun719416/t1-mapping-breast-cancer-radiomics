from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

# =========================
# 全局设置（投稿版）
# =========================
PREVALENCE = 0.296
TEST_N_BINS = 4
TRAIN_N_BINS = 10
Z_95 = 1.96

SAVE_DPI = 300
SAVE_PDF = True
SAVE_PNG = True

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 文件路径：改成你本机的
# =========================
train_files = {
    "Conventional": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_regular.xlsx",
    "Conventional + Mapping": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_regular+map.xlsx",
    "Fusion": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_fusion.xlsx",
    "Clinical": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_clinical_scores.xlsx",
}

test_files = {
    "Conventional": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular.xlsx",
    "Conventional + Mapping": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular+map.xlsx",
    "Fusion": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_fusion.xlsx",
    "Clinical": r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_clinical_scores.xlsx",
}

out_dir = Path(r"C:\Users\Sun\Desktop\3dslicer_malignant_nii\ROC_Figures\Calibration_Figures")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 工具函数
# =========================
def to_binary_label_pos1(s: pd.Series) -> np.ndarray:
    num = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    y = np.full_like(num, np.nan, dtype=float)
    m = ~np.isnan(num)
    y[m] = (num[m] == 1.0).astype(float)
    return y

def get_score_column(df: pd.DataFrame) -> str:
    candidates = ["PredProb_Cal", "CalibratedScore", "RadiomicsScore", "ClinicalScore",
                  "Score", "Probability", "PredProb", "Prob"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['patientid', 'label', 'id']]
    if numeric_cols:
        return numeric_cols[0]
    raise KeyError(f"缺少预测概率列。现有列：{df.columns.tolist()}")

def wilson_ci(k, n, z=Z_95):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def load_xy_from_excel(path: str):
    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(str(pth))

    df = pd.read_excel(pth, engine="openpyxl")
    if "Label" not in df.columns:
        raise KeyError(f"{pth.name} 缺少 Label 列")

    score_col = get_score_column(df)
    y = to_binary_label_pos1(df["Label"])
    p = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 0, 1)

    if len(y) == 0:
        raise ValueError(f"{pth.name} 无有效数据")
    return y, p

def bin_stats_equal_count(y_true, y_prob, n_bins):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0, 1)

    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)

    confs, accs, ns, lows, highs = [], [], [], [], []
    for idx in bins:
        nk = len(idx)
        if nk == 0:
            continue
        conf = y_prob[idx].mean()
        pos = int(y_true[idx].sum())
        acc = pos / nk
        lo, hi = wilson_ci(pos, nk)
        confs.append(conf); accs.append(acc); ns.append(nk)
        lows.append(lo); highs.append(hi)

    return (np.array(confs), np.array(accs), np.array(ns),
            np.array(lows), np.array(highs))

def bin_stats_unique_levels(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0, 1)

    df = pd.DataFrame({"y": y_true, "p": y_prob})
    g = df.groupby("p", sort=True)

    conf = g["p"].mean().to_numpy()
    acc  = g["y"].mean().to_numpy()
    n    = g.size().to_numpy()
    pos  = g["y"].sum().to_numpy()

    lo = np.zeros_like(acc, dtype=float)
    hi = np.zeros_like(acc, dtype=float)
    for i, (k, ni) in enumerate(zip(pos, n)):
        lo[i], hi[i] = wilson_ci(int(k), int(ni))

    return conf, acc, n, lo, hi

def ece_from_groups(acc, conf, n):
    n = np.asarray(n, dtype=float)
    w = n / n.sum()
    return float(np.sum(w * np.abs(acc - conf)))

def is_discrete_probability(y_prob, max_unique=8):
    uniq = pd.Series(y_prob).round(10).unique()
    return len(uniq) <= max_unique

# =========================
# 绘图：投稿版 2×2 面板
# =========================
def plot_calibration_panel_final(files_dict, cohort_title: str, out_dir: Path,
                                 n_bins_nonclinical: int,
                                 show_ci: bool,
                                 filename_stem: str,
                                 cohort_kind: str):
    """
    cohort_kind: "train" or "test"
    修改点：
      - test + clinical：只画点+CI，不画step、不连线（方案A）
      - train + clinical：保留轻量step（where="mid")
    """
    model_order = ["Conventional", "Conventional + Mapping", "Fusion", "Clinical"]
    panel_letters = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, model_name in enumerate(model_order):
        ax = axes[i]
        color = f"C{i}"

        # perfect line
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.6, zorder=0)

        # load
        y, p = load_xy_from_excel(files_dict[model_name])

        # metrics
        brier = brier_score_loss(y, p)

        clinical_like = (model_name.strip().lower() == "clinical") or is_discrete_probability(p, max_unique=8)

        if clinical_like:
            conf, acc, ns, lo, hi = bin_stats_unique_levels(y, p)
            ece = ece_from_groups(acc, conf, ns)

            if show_ci:
                yerr = np.vstack([acc - lo, hi - acc])
                yerr = np.maximum(yerr, 0)  # Ensure non-negative error bars
                # ★ 方案A：test的Clinical只画点（fmt="o"），不连线
                ax.errorbar(conf, acc, yerr=yerr, fmt="o", color=color,
                            markersize=5, capsize=3, elinewidth=1.4, alpha=0.95, zorder=3)
            else:
                ax.plot(conf, acc, "o", color=color, markersize=5, alpha=0.95, zorder=3)

            # train允许step；test不画step
            # if cohort_kind != "test":
            #     ax.step(conf, acc, where="mid", color=color, linewidth=2.0, alpha=0.95)

        else:
            # ===== 非Clinical：等人数分箱 =====
            conf, acc, ns, lo, hi = bin_stats_equal_count(y, p, n_bins=n_bins_nonclinical)
            ece = ece_from_groups(acc, conf, ns)

            # 非Clinical模型不画CI
            # if show_ci:
            #     yerr = np.vstack([acc - lo, hi - acc])
            #     yerr = np.maximum(yerr, 0)  # Ensure non-negative error bars
            #     ax.errorbar(conf, acc, yerr=yerr, fmt="none", color=color,
            #                 capsize=3, elinewidth=1.4, alpha=0.95, zorder=2)

            ax.plot(conf, acc, "o-", color=color, linewidth=2.0, markersize=5, alpha=0.95, zorder=3)

        # 标注 Brier / ECE
        ax.text(0.95, 0.05, f"Brier={brier:.3f}\nECE={ece:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"({panel_letters[i]}) {model_name}", loc="left")

    # 轴标签
    for ax in axes:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Predicted Probability")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Observed Fraction")

    fig.suptitle(f"{cohort_title} (Prevalence={PREVALENCE:.3f})", fontsize=14, y=0.99)
    plt.tight_layout()

    out_file = out_dir / f"{filename_stem}.png"
    plt.savefig(out_file, dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(out_dir / f"{filename_stem}.pdf", bbox_inches="tight")
    print(f"已保存：{out_file}")
    plt.close(fig)

# =========================
# 运行
# =========================
if __name__ == "__main__":
    plot_calibration_panel_final(
        train_files, "Train Cohort", out_dir,
        n_bins_nonclinical=TRAIN_N_BINS, show_ci=True,
        filename_stem="calibration_train_panel", cohort_kind="train"
    )

    plot_calibration_panel_final(
        test_files, "Test Cohort", out_dir,
        n_bins_nonclinical=TEST_N_BINS, show_ci=True,  # Test集默认也给CI
        filename_stem="calibration_test_panel", cohort_kind="test"
    )
