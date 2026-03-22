import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# IDI计算函数
def calculate_idi(model1_probs, model2_probs, y_true):
    """
    计算两个模型之间的IDI（综合分类改进指数）

    参数:
    model1_probs: 模型1的预测概率
    model2_probs: 模型2的预测概率
    y_true: 真实标签

    返回:
    idi: IDI值
    """
    # 正例和负例的索引
    pos_idx = y_true == 1
    neg_idx = y_true == 0

    # 计算两个模型在正负例上的平均预测概率
    p1_pos = np.mean(model1_probs[pos_idx]) if np.sum(pos_idx) > 0 else 0
    p1_neg = np.mean(model1_probs[neg_idx]) if np.sum(neg_idx) > 0 else 0
    p2_pos = np.mean(model2_probs[pos_idx]) if np.sum(pos_idx) > 0 else 0
    p2_neg = np.mean(model2_probs[neg_idx]) if np.sum(neg_idx) > 0 else 0

    # 计算IDI: (p2_pos - p2_neg) - (p1_pos - p1_neg)
    idi = (p2_pos - p2_neg) - (p1_pos - p1_neg)

    return idi


# 定义NRI计算函数
def calculate_nri(model1_probs, model2_probs, y_true, threshold=0.5):
    """
    计算两个模型之间的NRI（净重新分类改善指数）

    参数:
    model1_probs: 模型1的预测概率
    model2_probs: 模型2的预测概率
    y_true: 真实标签
    threshold: 分类阈值，默认为0.5

    返回:
    total_nri: 总NRI值
    event_nri: 事件组NRI
    nonevent_nri: 非事件组NRI
    """
    # 根据阈值将概率转换为预测标签
    model1_preds = (model1_probs >= threshold).astype(int)
    model2_preds = (model2_probs >= threshold).astype(int)

    # 事件组（y=1）的净重分类
    event_up = np.sum((model2_preds > model1_preds) & (y_true == 1))
    event_down = np.sum((model2_preds < model1_preds) & (y_true == 1))
    n_events = np.sum(y_true == 1)
    event_nri = (event_up - event_down) / n_events if n_events > 0 else 0

    # 非事件组（y=0）的净重分类
    nonevent_up = np.sum((model2_preds > model1_preds) & (y_true == 0))
    nonevent_down = np.sum((model2_preds < model1_preds) & (y_true == 0))
    n_nonevents = np.sum(y_true == 0)
    nonevent_nri = (nonevent_down - nonevent_up) / n_nonevents if n_nonevents > 0 else 0

    # 总NRI
    total_nri = event_nri + nonevent_nri

    return total_nri, event_nri, nonevent_nri


# Bootstrap置信区间计算函数
def bootstrap_metric(model1_probs, model2_probs, y_true, metric_func, n_bootstrap=1000, **kwargs):
    """
    使用Bootstrap计算指标的置信区间和p值

    参数:
    model1_probs: 模型1的预测概率
    model2_probs: 模型2的预测概率
    y_true: 真实标签
    metric_func: 指标计算函数
    n_bootstrap: Bootstrap次数
    **kwargs: 传递给metric_func的额外参数

    返回:
    metric_values: Bootstrap得到的指标值列表
    ci: 置信区间 (lower, upper)
    p_value: p值
    """
    n_samples = len(y_true)
    metric_values = []

    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        metric_val = metric_func(
            model1_probs.iloc[indices] if hasattr(model1_probs, 'iloc') else model1_probs[indices],
            model2_probs.iloc[indices] if hasattr(model2_probs, 'iloc') else model2_probs[indices],
            y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices],
            **kwargs
        )
        # 如果返回多个值，只取第一个（总指标）
        if isinstance(metric_val, tuple):
            metric_val = metric_val[0]
        metric_values.append(metric_val)

    ci_lower = np.percentile(metric_values, 2.5)
    ci_upper = np.percentile(metric_values, 97.5)

    # 计算p值（单侧检验，检验指标是否大于0）
    p_value = np.mean(np.array(metric_values) <= 0)

    return metric_values, (ci_lower, ci_upper), p_value


def main():
    """主函数"""
    try:
        # 读取数据
        print("正在读取数据...")
        regular_radiomics_df = pd.read_excel(
            r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular.xlsx")
        regular_map_radiomics_df = pd.read_excel(
            r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular+map.xlsx")
        fusion_radiomics_df = pd.read_excel(
            r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_fusion.xlsx")
        # 读取临床评分表（自动检测评分列）
        clinical_path = r"C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_clinical_scores.xlsx"
        clinical_df = pd.read_excel(clinical_path)

        print(
            f"数据读取完成: Conventional({len(regular_radiomics_df)}), Conventional+Map({len(regular_map_radiomics_df)}), Fusion({len(fusion_radiomics_df)}), Clinical({len(clinical_df)})")

        # 合并数据（先合并三个影像学得分，再合并临床得分）
        merged_df_regular = pd.merge(regular_radiomics_df[['PatientID', 'RadiomicsScore', 'Label']],
                                     regular_map_radiomics_df[['PatientID', 'RadiomicsScore']],
                                     on='PatientID', suffixes=('_regular', '_regular_map'))

        merged_df_fusion = pd.merge(merged_df_regular, fusion_radiomics_df[['PatientID', 'RadiomicsScore']],
                                    on='PatientID', suffixes=('_regular_map', '_fusion'))

        # 尝试识别临床评分的列名，常见候选
        score_col = None
        candidates = ['RadiomicsScore', 'ClinicalScore', 'Score', 'Probability', 'PredictedProbability', 'Score_clinical']
        for c in candidates:
            if c in clinical_df.columns:
                score_col = c
                break
        if score_col is None:
            # 选择第一个数值列（排除ID和Label）
            numeric_cols = clinical_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c.lower() not in ['patientid', 'label']]
            if len(numeric_cols) > 0:
                score_col = numeric_cols[0]
            else:
                raise ValueError(f"无法在临床表中找到合适的评分列，clinical columns: {clinical_df.columns.tolist()}")

        # 将临床评分列重命名为 ClinicalScore
        clinical_df = clinical_df.rename(columns={score_col: 'ClinicalScore'})

        # 合并临床评分
        merged_df_all = pd.merge(merged_df_fusion, clinical_df[['PatientID', 'ClinicalScore']], on='PatientID')

        # 数据基本信息
        print(f"\n合并后数据形状: {merged_df_all.shape}")
        print(f"正例数量: {sum(merged_df_all['Label'] == 1)}")
        print(f"负例数量: {sum(merged_df_all['Label'] == 0)}")

        # 提取模型的概率值和真实标签
        model_probs_regular = merged_df_all['RadiomicsScore_regular']
        model_probs_regular_map = merged_df_all['RadiomicsScore_regular_map']
        model_probs_fusion = merged_df_all['RadiomicsScore']
        model_probs_clinical = merged_df_all['ClinicalScore']
        y_true = merged_df_all['Label']

        # 模型列表（包含临床）
        # 顺序修改：Clinical -> Conventional -> Conventional + Map -> Fusion
        model_probs = [model_probs_clinical, model_probs_regular, model_probs_regular_map, model_probs_fusion]
        # 修改点：更改名称以在图表轴上显示为 Conventional 和 Conventional + Map
        model_names = ['Clinical', 'Conventional', 'Conventional + Map', 'Fusion']
        n_models = len(model_names)

        print("\n" + "=" * 50)
        print("IDI 分析结果")
        print("=" * 50)

        # 初始化IDI矩阵和置信区间矩阵
        idi_matrix = np.zeros((n_models, n_models))
        idi_ci_matrix = np.zeros((n_models, n_models, 2))  # 存储置信区间
        idi_p_matrix = np.zeros((n_models, n_models))  # 存储p值

        # 计算IDI矩阵
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    idi_val = calculate_idi(model_probs[j], model_probs[i], y_true)
                    idi_matrix[i, j] = idi_val

                    # Bootstrap计算置信区间
                    _, ci, p_value = bootstrap_metric(model_probs[j], model_probs[i], y_true, calculate_idi)
                    idi_ci_matrix[i, j] = ci
                    idi_p_matrix[i, j] = p_value

        # 输出IDI结果（动态打印列数）
        print("\nIDI Matrix (行→列):")
        # header
        header = "        " + "  ".join([f"{name:>10}" for name in model_names])
        print(header)
        for i, name in enumerate(model_names):
            row_vals = "  ".join([f"{idi_matrix[i, j]:.4f}" for j in range(n_models)])
            print(f"{name:10} {row_vals}")

        print("\nIDI 置信区间 (95%):")
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    print(f"{model_names[j]} → {model_names[i]}: {idi_matrix[i, j]:.4f} "
                          f"(95% CI: [{idi_ci_matrix[i, j, 0]:.4f}, {idi_ci_matrix[i, j, 1]:.4f}], "
                          f"p = {idi_p_matrix[i, j]:.4f})")

        # 绘制IDI热图
        plt.figure(figsize=(2 + n_models * 2.5, 2 + n_models * 2.0))
        mask = np.eye(n_models, dtype=bool)  # 掩码对角线
        sns.heatmap(idi_matrix, annot=True, fmt=".3f", xticklabels=model_names,
                    yticklabels=model_names, cmap="RdBu_r", center=0, mask=mask, annot_kws={"size": 12})
        plt.title('Integrated Discrimination Improvement (IDI) Heatmap', fontsize=14)
        plt.xlabel('Reference Model (Old model)', fontsize=12)
        plt.ylabel('Comparison Model (New model)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 保存IDI热图
        plt.savefig('C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/IDI_test_heatmap.png', dpi=300,
                    bbox_inches='tight')
        plt.show()
        print("\n" + "=" * 50)
        print("NRI 分析结果")
        print("=" * 50)

        # 初始化NRI矩阵
        nri_matrix = np.zeros((n_models, n_models))
        nri_ci_matrix = np.zeros((n_models, n_models, 2))
        nri_p_matrix = np.zeros((n_models, n_models))

        # 计算NRI矩阵
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    nri_val, event_nri, nonevent_nri = calculate_nri(model_probs[j], model_probs[i], y_true)
                    nri_matrix[i, j] = nri_val

                    # Bootstrap计算置信区间
                    _, ci, p_value = bootstrap_metric(model_probs[j], model_probs[i], y_true, calculate_nri)
                    nri_ci_matrix[i, j] = ci
                    nri_p_matrix[i, j] = p_value

        # 输出NRI结果（动态打印列数）
        print("\nNRI Matrix (行→列):")
        header = "        " + "  ".join([f"{name:>10}" for name in model_names])
        print(header)
        for i, name in enumerate(model_names):
            row_vals = "  ".join([f"{nri_matrix[i, j]:.4f}" for j in range(n_models)])
            print(f"{name:10} {row_vals}")

        print("\nNRI 置信区间 (95%):")
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    print(f"{model_names[j]} → {model_names[i]}: {nri_matrix[i, j]:.4f} "
                          f"(95% CI: [{nri_ci_matrix[i, j, 0]:.4f}, {nri_ci_matrix[i, j, 1]:.4f}], "
                          f"p = {nri_p_matrix[i, j]:.4f})")

        # 绘制NRI热图
        plt.figure(figsize=(2 + n_models * 2.5, 2 + n_models * 2.0))
        sns.heatmap(nri_matrix, annot=True, fmt=".3f", xticklabels=model_names,
                    yticklabels=model_names, cmap="RdBu_r", center=0, mask=mask,
                    vmin=-1, vmax=1, annot_kws={"size": 12})
        plt.title('Net Reclassification Improvement (NRI) Heatmap', fontsize=14)
        plt.xlabel('Reference Model (Old model)', fontsize=12)
        plt.ylabel('Comparison Model (New model)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 保存NRI热图
        plt.savefig('C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/NRI_test_heatmap.png', dpi=300,
                    bbox_inches='tight')
        plt.show()

        # 综合结果总结：打印所有成对比较
        print("\n" + "=" * 50)
        print("综合结果总结")
        print("=" * 50)

        # 指定需要展示的特定比较 (Reference/Old Index, Comparison/New Index, Description)
        # Indices: 0=Clinical, 1=Conventional, 2=Conventional+Map, 3=Fusion
        # 修改点：文本描述也同步更新
        specific_comparisons = [
            (0, 3, "Clinical → Fusion"),
            (1, 2, "Conventional → Conventional + Map"),
            (2, 3, "Conventional + Map → Fusion"),
            (1, 3, "Conventional → Fusion"),
            (0, 1, "Clinical → Conventional"),
            (0, 2, "Clinical → Conventional + Map")
        ]

        for ref_idx, comp_idx, desc in specific_comparisons:
            # i is New (Comparison), j is Old (Reference)
            # idi_matrix[i, j] stores metric for New vs Old
            i, j = comp_idx, ref_idx

            idi_val = idi_matrix[i, j]
            idi_p = idi_p_matrix[i, j]
            nri_val = nri_matrix[i, j]
            nri_p = nri_p_matrix[i, j]

            print(f"\n{desc}:")
            print(
                f"  IDI: {idi_val:.4f} (p = {idi_p:.4f}) {'***' if idi_p < 0.001 else '**' if idi_p < 0.01 else '*' if idi_p < 0.05 else 'ns'}")
            print(
                f"  NRI: {nri_val:.4f} (p = {nri_p:.4f}) {'***' if nri_p < 0.001 else '**' if nri_p < 0.01 else '*' if nri_p < 0.05 else 'ns'}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()