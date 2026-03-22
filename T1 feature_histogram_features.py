import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import skew, kurtosis, entropy
from nilearn.image import resample_to_img

print("所有库导入成功。")


def compute_first_order_features(values, nbins=32, bin_width=None, ignore_zeros=False):
   
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    if ignore_zeros:
        vals = vals[vals != 0]

    if vals.size == 0:
        nan_dict = {
            "Mean": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan, "Range": np.nan,
            "Variance": np.nan, "Std": np.nan, "RMS": np.nan, "MAD": np.nan, "RobustMAD": np.nan,
            "Skewness": np.nan, "Kurtosis": np.nan, "P10": np.nan, "P25": np.nan, "P50": np.nan,
            "P75": np.nan, "P90": np.nan, "Energy": np.nan, "Entropy": np.nan
        }
        return nan_dict

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


    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])

    if bin_width is not None and bin_width > 0:
        bins = np.arange(vmin, vmax + bin_width, bin_width)
        if bins.size < 2:
            bins = np.linspace(vmin, vmax, max(2, nbins))
    else:
        bins = nbins

    counts, _ = np.histogram(vals, bins=bins)
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    probs = probs[probs > 0] 

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

def extract_features_for_all_patients(base_dir, output_filename="histogram_features.xlsx"):
  
    print("="*60)
    print("|| 开始提取直方图特征".ljust(58) + "||")
    print("="*60)
    print(f"数据根目录: {base_dir}\n")

    if not os.path.isdir(base_dir):
        print(f"错误: 目录 '{base_dir}' 不存在。请检查路径。")
        return

    all_features = []

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

            image_files = glob.glob(os.path.join(sequence_path, "*.nii"))
            mask_files = glob.glob(os.path.join(sequence_path, "*.nii.gz"))

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
  
                image_nii = nib.load(image_path)
                mask_nii = nib.load(mask_path)

                if image_nii.shape != mask_nii.shape:
                    print(f"  [信息] 图像和掩膜的尺寸不匹配，正在自动重采样掩膜...")
                    print(f"     原始图像尺寸: {image_nii.shape}")
                    print(f"     原始掩膜尺寸: {mask_nii.shape}")
                    try:
                        mask_nii = resample_to_img(mask_nii, image_nii, interpolation='nearest', force_resample=True, copy_header=True)
                        print(f"     重采样后掩膜尺寸: {mask_nii.shape}")
                    except Exception as e:
                        print(f"  [失败] 重采样失败: {e}")
                        continue

   
                try:
  
                    image_data = image_nii.get_fdata()
                    mask_data = mask_nii.get_fdata()

     
                    all_feats = {}

                    roi_voxels = image_data[mask_data > 0]
                    if roi_voxels.size == 0:
                        print("  [警告] 掩膜为空，无法提取特征。")
                        continue
                    orig_feats = compute_first_order_features(roi_voxels, ignore_zeros=True)
                    orig_feats_pref = {f"original_{k}": v for k, v in orig_feats.items()}
                    all_feats.update(orig_feats_pref)

                    wavelet_extracted = False
                    try:
                        import pywt
                        from skimage.transform import resize
                        coeffs = pywt.dwtn(image_data, wavelet='coif1', mode='periodization')
                        for key, arr in coeffs.items():
                            try:
                            
                                mask_downsampled = resize(
                                    mask_data,
                                    arr.shape,
                                    order=0,
                                    anti_aliasing=False,
                                    preserve_range=True
                                ).astype(bool)

                                coeff_voxels = arr[mask_downsampled]

                                if coeff_voxels.size == 0:
                                    wfeat = {k: np.nan for k in orig_feats.keys()}
                                else:
                                    wfeat = compute_first_order_features(coeff_voxels, ignore_zeros=True)
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

    df = pd.DataFrame(all_features)
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
    print("\n" + "="*60)
    print("|| 开始添加标签并拆分表格".ljust(58) + "||")
    print("="*60)

    label_path = r"C:\Users\Sun\Desktop\resampled2\patient_label.xlsx"
    if not os.path.exists(label_path):
        print(f"错误: 标签文件 '{label_path}' 不存在。请确保标签文件位于数据根目录中。")
        return

    try:
        label_df = pd.read_excel(label_path)
        label_df['PatientID'] = label_df['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()
        print("检查 'patient-label.xlsx' 中的 PatientID 列（去除空格并统一为小写字母）:")
        print(label_df['PatientID'].head())
    except Exception as e:
        print(f"错误: 读取标签文件失败: {e}")
        return

    df['PatientID'] = df['PatientID'].str.replace(r'\s+', '', regex=True).str.lower()

    label_dict = dict(zip(label_df['PatientID'], label_df['Label']))

    df['Label'] = df['PatientID'].map(label_dict)

    unmatched_patient_ids = df[df['Label'].isna()]['PatientID'].unique()
    if len(unmatched_patient_ids) > 0:
        print(f"以下 PatientID 没有找到匹配的标签: {unmatched_patient_ids}")

    cols = [col for col in df.columns if col != 'Label']
    cols.insert(1, 'Label') 
    df = df[cols]

    merged_output_path = os.path.join(base_dir, "merged_histogram_features_with_labels_reader2.xlsx")
    df.to_excel(merged_output_path, index=False)
    print(f"合并完成，已保存为 '{merged_output_path}'.")

    sequences = ['mappre', 'mappost']
    split_data = {seq: df[df['Sequence'] == seq] for seq in sequences}

    split_output_paths = {
        'mappre': os.path.join(base_dir, "mappre_reader2.xlsx"),
        'mappost': os.path.join(base_dir, "mappost_reader2.xlsx")
    }

    for seq, split_df in split_data.items():
        split_df.to_excel(split_output_paths[seq], index=False)
        print(f"拆分完成，{seq} 表格已保存为 '{split_output_paths[seq]}'.")

    print("\n" + "="*60)
    print("|| 添加标签并拆分表格完成！".ljust(58) + "||")
    print("="*60)


    print("\n" + "="*60)
    print("|| 开始计算 Delta 特征".ljust(58) + "||")
    print("="*60)
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

    mappre_df.columns = mappre_df.columns.str.strip()
    mappost_df.columns = mappost_df.columns.str.strip()

    columns_to_remove = ["PatientID", "Label", "Sequence"]
    mappre_df_cleaned = mappre_df.drop(columns=columns_to_remove)
    mappost_df_cleaned = mappost_df.drop(columns=columns_to_remove)

    common_features = mappre_df_cleaned.columns.intersection(mappost_df_cleaned.columns)

    delta1_df = mappre_df_cleaned[common_features] - mappost_df_cleaned[common_features]

    delta2_df = delta1_df / mappre_df_cleaned[common_features]
    delta1_df = pd.concat([mappre_df[["PatientID", "Label", "Sequence"]], delta1_df], axis=1)
    delta2_df = pd.concat([mappre_df[["PatientID", "Label", "Sequence"]], delta2_df], axis=1)

    delta1_df["Sequence"] = "delta1"
    delta2_df["Sequence"] = "delta2"

    delta1_path = os.path.join(base_dir, "delta1_reader2.xlsx")
    delta2_path = os.path.join(base_dir, "delta2_reader2.xlsx")
    delta1_df.to_excel(delta1_path, index=False)
    delta2_df.to_excel(delta2_path, index=False)

    print(f"Delta1 表格已保存为 '{delta1_path}'.")
    print(f"Delta2 表格已保存为 '{delta2_path}'.")

    print("\nDelta1 DataFrame:")
    print(delta1_df.head())

    print("\nDelta2 DataFrame:")
    print(delta2_df.head())

    print("\n" + "="*60)
    print("|| 计算 Delta 特征完成！".ljust(58) + "||")
    print("="*60)

if __name__ == "__main__":
    print("Entering main block")

    BASE_DIRECTORY = r"C:\Users\Sun\Desktop\resampled2"
    extract_features_for_all_patients(BASE_DIRECTORY)
