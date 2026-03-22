import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import skew, kurtosis, entropy
from nilearn.image import resample_to_img

print("所有库导入成功。")

# ==============================================================================
# ========================== 1. 特征计算函数 ====================================
# ==============================================================================

def compute_first_order_features(values, nbins=32, bin_width=None, ignore_zeros=False):
    """
    计算并返回常见的直方图（first-order）特征。

    参数:
    - values (np.ndarray): 从ROI中提取的1D体素强度数组。
    - nbins (int): 如果不使用固定箱宽，则指定直方图的箱数。
    - bin_width (float, optional): 如��指定，则使用固定箱宽进行离散化，覆盖nbins。
    - ignore_zeros (bool): 是否在计算前忽略值为0的体素。

    返回:
    - dict: 包含所有计算出的特征的字典。
    """
    # 清理数据，移除无穷大或NaN值
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    if ignore_zeros:
        vals = vals[vals != 0]

    # 如果没有有效体���，返回NaN
    if vals.size == 0:
        nan_dict = {
            "Mean": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan, "Range": np.nan,
            "Variance": np.nan, "Std": np.nan, "RMS": np.nan, "MAD": np.nan, "RobustMAD": np.nan,
            "Skewness": np.nan, "Kurtosis": np.nan, "P10": np.nan, "P25": np.nan, "P50": np.nan,
            "P75": np.nan, "P90": np.nan, "Energy": np.nan, "Entropy": np.nan
        }
        return nan_dict

    # --- 基本统计特征 ---
    mean = vals.mean()
    median = np.median(vals)
    vmin = vals.min()
    vmax = vals.max()
    range_val = vmax - vmin
    variance = vals.var(ddof=1) if vals.size > 1 else 0.0
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(vals**2))
    mad = np.mean(np.abs(vals - mean))
    robust_mad = np.median(np.abs(vals - np.median(vals)))
    skewness = float(skew(vals, bias=False)) if vals.size > 2 else np.nan
    kurtosis_val = float(kurtosis(vals, fisher=True, bias=False)) if vals.size > 3 else np.nan

    # --- 百分位数特征 ---
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])

    # --- 基于直方图的特征 ---
    # 离散化
    if bin_width is not None and bin_width > 0:
        bins = np.arange(vmin, vmax + bin_width, bin_width)
        if bins.size < 2:
            bins = np.linspace(vmin, vmax, max(2, nbins))
    else:
        bins = nbins

    counts, _ = np.histogram(vals, bins=bins)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    probs = probs[probs > 0] # 只考虑非零概率

    # 能量、熵、均匀性
    energy = np.sum(vals**2)
    entropy_val = float(entropy(probs, base=2)) if probs.size > 0 else 0.0
    uniformity = float(np.sum(probs**2))

    features = {
        "Mean": float(mean),
        "Median": float(median),
        "Min": float(vmin),
        "Max": float(vmax),
        "Range": float(range_val),
        "Variance": float(variance),
        "Std": float(std),
        "RMS": float(rms),
        "MAD": float(mad),
        "RobustMAD": float(robust_mad),
        "Skewness": skewness,
        "Kurtosis": kurtosis_val,
        "P10": float(p10),
        "P25": float(p25),
        "P50": float(p50),
        "P75": float(p75),
        "P90": float(p90),
        "Energy": float(energy),
        "Entropy": entropy_val
    }
    return features

# ==============================================================================
# ========================== 2. 主流程 =========================================
# ==============================================================================

def extract_features_for_all_patients(base_dir, output_filename="histogram_features.xlsx"):
    """
    遍历所有病人文件夹，提取mappre和mappost的直方图特征，并保存到Excel。
    """
    print("="*60)
    print("|| 开始提取直方图特征".ljust(58) + "||")
    print("="*60)
    print(f"数据根目录: {base_dir}\n")

    # 检查根目录是否存在
    if not os.path.isdir(base_dir):
        print(f"错误: 目录 '{base_dir}' 不存在。请检查路径。")
        return

    all_features = []

    # 获取所有病人文件夹
    patient_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    if not patient_dirs:
        print("错误: 在根目录下没有找到任何病人文件夹。")
        return

    print(f"发现 {len(patient_dirs)} 个病人文件夹。")

    for patient_id in patient_dirs:
        print(f"\n--- 正在处理病人: {patient_id} ---")
        patient_path = os.path.join(base_dir, patient_id)

        for sequence_name in ["mappre", "mappost"]:
            sequence_path = os.path.join(patient_path, sequence_name)

            if not os.path.isdir(sequence_path):
                print(f"  [跳过] 序列文件夹 '{sequence_name}' 不存在。")
                continue

            # 查找图像 (.nii) 和掩膜 (.nii.gz) 文件
            image_files = glob.glob(os.path.join(sequence_path, "*.nii"))
            mask_files = glob.glob(os.path.join(sequence_path, "*.nii.gz"))

            # 校验文件是否存在
            if not image_files:
                print(f"  [警告] 在 '{sequence_path}' 中未找到图像文件 (.nii)。")
                continue
            if not mask_files:
                print(f"  [警告] 在 '{sequence_path}' 中未找到掩膜文件 (.nii.gz)。")
                continue

            image_path = image_files[0]
            mask_path = mask_files[0]
            print(f"  -> 正在处理序列: {sequence_name}")
            print(f"     图像: {os.path.basename(image_path)}")
            print(f"     掩膜: {os.path.basename(mask_path)}")

            try:
                # 加载图像和掩膜
                image_nii = nib.load(image_path)
                mask_nii = nib.load(mask_path)

                # 检查图像和掩膜的尺寸是否匹配
                if image_nii.shape != mask_nii.shape:
                    print(f"  [信息] 图像和掩膜的尺寸不匹配，正在自动重采样掩膜...")
                    print(f"     原始图像尺寸: {image_nii.shape}")
                    print(f"     原始掩膜尺寸: {mask_nii.shape}")
                    try:
                        # 使用 Nilearn 重采样掩膜到图像空间
                        mask_nii = resample_to_img(mask_nii, image_nii, interpolation='nearest', force_resample=True, copy_header=True)
                        print(f"     重采样后掩膜尺寸: {mask_nii.shape}")
                    except Exception as e:
                        print(f"  [失败] 重采样失败: {e}")
                        continue

                # 手动计算直方图特征
                try:
                    # 提取图像和掩膜中的强度值
                    image_data = image_nii.get_fdata()
                    mask_data = mask_nii.get_fdata()

                    # 汇总所有特征（包含原始与小波子带）
                    all_feats = {}

                    # 计算原始图像的直方图特征（并添加前缀 original_ ）
                    roi_voxels = image_data[mask_data > 0]
                    if roi_voxels.size == 0:
                        print("  [警告] 掩膜为空，无法提取特征。")
                        continue
                    orig_feats = compute_first_order_features(roi_voxels, ignore_zeros=True)
                    orig_feats_pref = {f"original_{k}": v for k, v in orig_feats.items()}
                    all_feats.update(orig_feats_pref)

                    # =========== 小波变换（单层3D小波） ===========
                    wavelet_extracted = False
                    try:
                        import pywt
                        from skimage.transform import resize

                        # 使用单层3D离散小波（例如 coif1），mode 使用 periodization 保持边界稳定
                        coeffs = pywt.dwtn(image_data, wavelet='coif1', mode='periodization')
                        # coeffs 是一个 dict，键例如 ('aaa','aad',... ) 或 'aaa' 等，取决于 pywt 版本
                        for key, arr in coeffs.items():
                            try:
                                # !!! 关键修复：将掩膜下采样到与小波子带相同的尺寸 !!!
                                # 使用 order=0 (最近邻插值) 保持掩膜的二值特性
                                mask_downsampled = resize(
                                    mask_data,
                                    arr.shape,
                                    order=0,
                                    anti_aliasing=False,
                                    preserve_range=True
                                ).astype(bool)

                                coeff_voxels = arr[mask_downsampled]

                                if coeff_voxels.size == 0:
                                    # 若该子带在ROI内无体素，则填充 NaN 特征
                                    wfeat = {k: np.nan for k in orig_feats.keys()}
                                else:
                                    wfeat = compute_first_order_features(coeff_voxels, ignore_zeros=True)
                                # 前缀为 wavelet-<key>_
                                pref = f"wavelet-{key}_"
                                wfeat_pref = {pref + k: v for k, v in wfeat.items()}
                                all_feats.update(wfeat_pref)
                            except Exception as ee:
                                print(f"    [警告] 小波子带 {key} 特征提取失败: {ee}")
                        wavelet_extracted = True
                    except ImportError:
                        print("  [提示] pywt (PyWavelets) 或 scikit-image 未安装，跳过小波特征提取。可通过 'pip install pywt scikit-image' 安装。")
                    except Exception as e:
                        print(f"  [警告] 小波分解失败: {e}")

                    # 添加标识信息
                    all_feats['PatientID'] = patient_id
                    all_feats['Sequence'] = sequence_name

                    all_features.append(all_feats)
                    feature_count = len(all_feats) - 2
                    if wavelet_extracted:
                        print(f"  [成功] 提取了 {feature_count} 个特征（含原始与小波子带）。")
                    else:
                        print(f"  [成功] 提取了 {feature_count} 个特征（仅原始）。")

                except Exception as e:
                    print(f"  [失败] 特征计算失败: {e}")

            except Exception as e:
                print(f"  [失败] 处理文件时发生错误: {e}")

    if not all_features:
        print("\n未能提取任何特征。请检查文件结构和内���。")
        return

    # --- 保存结果到Excel ---
    df = pd.DataFrame(all_features)

    # 调整列顺序，将ID和序列名放在最前面
    cols = ['PatientID', 'Sequence'] + [col for col in df.columns if col not in ['PatientID', 'Sequence']]
    df = df[cols]

    output_path = os.path.join(base_dir, output_filename)
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print("\n" + "="*60)
        print("|| 特征提取完成！".ljust(58) + "||")
        print("="*60)
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"\n错误: 保存Excel文件失败: {e}")
        return

    # ==============================================================================
    # ========================== 3. 添加标签并拆分表格 ===============================
    # ==============================================================================

    print("\n" + "="*60)
    print("|| 开始添加标签并拆分表格".ljust(58) + "||")
    print("="*60)

    # 读取标签文件（假设标签文件在 base_dir 中）
    label_path = r"C:\Users\Sun\Desktop\resampled2\patient_label.xlsx"
    if not os.path.exists(label_path):
        print(f"错误: 标签文件 '{label_path}' 不存在。请确保标签文件位于数据根目录中。")
        return

    try:
        label_df = pd.read_excel(label_path)
        # 去除空格并统一大小写，确保格式一致
        label_df['PatientID'] = label_df['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()
        print("检查 'patient-label.xlsx' 中的 PatientID 列（去除空格并统一为小写字母）:")
        print(label_df['PatientID'].head())
    except Exception as e:
        print(f"错误: 读取标签文件失败: {e}")
        return

    # 去除空格并统一大小写，确保 results_df 中 PatientID 列格式一致
    df['PatientID'] = df['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()

    # 将 label_df 中的 PatientID 和 Label 映射为字典
    label_dict = dict(zip(label_df['PatientID'], label_df['Label']))

    # 使用 map 函数根据 PatientID 填充 Label 列
    df['Label'] = df['PatientID'].map(label_dict)

    # 检查哪些 PatientID 没有匹配的标签
    unmatched_patient_ids = df[df['Label'].isna()]['PatientID'].unique()
    if len(unmatched_patient_ids) > 0:
        print(f"以下 PatientID 没有找到匹配的标签: {unmatched_patient_ids}")

    # 调整列的顺序，将 Label 列放到第二列
    cols = [col for col in df.columns if col != 'Label']
    cols.insert(1, 'Label')  # 在第二列插入 Label
    df = df[cols]

    # 将合并后的结果保存为新的Excel文件
    merged_output_path = os.path.join(base_dir, "merged_histogram_features_with_labels_reader2.xlsx")
    df.to_excel(merged_output_path, index=False)
    print(f"合并完成，已保存为 '{merged_output_path}'.")

    # 拆分表格为 mappre 和 mappost
    sequences = ['mappre', 'mappost']
    split_data = {seq: df[df['Sequence'] == seq] for seq in sequences}

    # 定义输出文件路径
    split_output_paths = {
        'mappre': os.path.join(base_dir, "mappre_reader2.xlsx"),
        'mappost': os.path.join(base_dir, "mappost_reader2.xlsx")
    }

    # 将每个 DataFrame 保存为单独的 Excel 文件
    for seq, split_df in split_data.items():
        split_df.to_excel(split_output_paths[seq], index=False)
        print(f"拆分完成，{seq} 表格已保存为 '{split_output_paths[seq]}'.")

    print("\n" + "="*60)
    print("|| 添加标签并拆分表格完成！".ljust(58) + "||")
    print("="*60)

    # ==============================================================================
    # ========================== 4. 计算 Delta 特征 ===============================
    # ==============================================================================

    print("\n" + "="*60)
    print("|| 开始计算 Delta 特征".ljust(58) + "||")
    print("="*60)

    # 读取 mappre 和 mappost 表格
    mappre_path = os.path.join(base_dir, "mappre_reader2.xlsx")
    mappost_path = os.path.join(base_dir, "mappost_reader2.xlsx")

    print(f"尝试读取 mappre 文件: {mappre_path}")
    print(f"尝试读取 mappost 文件: {mappost_path}")

    if not os.path.exists(mappre_path) or not os.path.exists(mappost_path):
        print(f"错误: 找不到 mappre 或 mappost 表格文件。")
        print(f"请检查第3步是否成功执行，并确保文件已保存到上述路径。")
        return

    try:
        mappre_df = pd.read_excel(mappre_path)
        mappost_df = pd.read_excel(mappost_path)
    except Exception as e:
        print(f"错误: 读取表格文件失败: {e}")
        return

    # 确保列名的一致性（去掉额外空格或大小写问题）
    mappre_df.columns = mappre_df.columns.str.strip()
    mappost_df.columns = mappost_df.columns.str.strip()

    # 删除不需要的列 "PatientID", "Label", "Sequence"
    columns_to_remove = ["PatientID", "Label", "Sequence"]
    mappre_df_cleaned = mappre_df.drop(columns=columns_to_remove)
    mappost_df_cleaned = mappost_df.drop(columns=columns_to_remove)

    # 找出两个数据集之间的共同特征
    common_features = mappre_df_cleaned.columns.intersection(mappost_df_cleaned.columns)

    # 计算共同特征的差值 (delta1)
    delta1_df = mappre_df_cleaned[common_features] - mappost_df_cleaned[common_features]

    # 新增计算 delta2: 差值除以 mappre 的特征值
    delta2_df = delta1_df / mappre_df_cleaned[common_features]

    # 将 "PatientID", "Label", "Sequence" 列添加回差异数据
    delta1_df = pd.concat([mappre_df[["PatientID", "Label", "Sequence"]], delta1_df], axis=1)
    delta2_df = pd.concat([mappre_df[["PatientID", "Label", "Sequence"]], delta2_df], axis=1)

    # 修改 "Sequence" 列的内容，将其替换为 "delta1" 和 "delta2"
    delta1_df["Sequence"] = "delta1"
    delta2_df["Sequence"] = "delta2"

    # 保存为 delta1.xlsx 和 delta2.xlsx 文件
    delta1_path = os.path.join(base_dir, "delta1_reader2.xlsx")
    delta2_path = os.path.join(base_dir, "delta2_reader2.xlsx")
    delta1_df.to_excel(delta1_path, index=False)
    delta2_df.to_excel(delta2_path, index=False)

    print(f"Delta1 表格已保存为 '{delta1_path}'.")
    print(f"Delta2 表格已保存为 '{delta2_path}'.")

    # 打印前几行数据以供查看
    print("\nDelta1 DataFrame:")
    print(delta1_df.head())

    print("\nDelta2 DataFrame:")
    print(delta2_df.head())

    print("\n" + "="*60)
    print("|| 计算 Delta 特征完成！".ljust(58) + "||")
    print("="*60)

if __name__ == "__main__":
    print("Entering main block")
    # !!! 重要: 请在这里设置您的数据根目录 !!!
    # 这是包含所有病人文件夹的顶级目录。
    BASE_DIRECTORY = r"C:\Users\Sun\Desktop\resampled2"

    # 运行主函数
    extract_features_for_all_patients(BASE_DIRECTORY)
