#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.executable)


# In[1]:


import radiomics
from radiomics import featureextractor
print(radiomics.__version__)


# In[1]:


#-------------------------------------------------原始+8个小波变换------------------------------------------------------------
import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ========= 基本路径与序列设置 =========
#dataDir = r"C:\Users\Sun\Desktop\3dslicer_tougao_malignant_nii"  # 数据存放路径
dataDir = r"C:\Users\Sun\Desktop\resampled2"
sequences = ['stir', 'mappre', 'mappost', 'DCE', 'ADC']

# 仅列出病人文件夹
folderList = [f for f in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, f))]

# ========= 定义特征提取器 =========
extractor = featureextractor.RadiomicsFeatureExtractor()  # 实例化

# 归一化设置（保留你原来的写法）
extractor.enableImageNormalization = True
extractor.imageNormalizingMethod = "z-score"

# 启用小波变换（新增）
# 默认使用 pyRadiomics 的小波配置（如 'coif1'），会对 8 个子带分别提特征
extractor.enableImageTypes(Wavelet={})

# （可选）也可以显式指定设置，确保生效与可复现性：
# extractor.settings.update({
#     'normalize': True,
#     'normalizeScale': 1,
#     'label': 1,  # 掩膜前景标签（如你的掩膜是 1）
#     # 'binWidth': 25,
#     # 'resampledPixelSpacing': [1, 1, 1],
# })

# ========= 结果表 =========
df = pd.DataFrame()

# ========= 遍历每个病人 =========
for folder in folderList:
    patient_features = []

    for seq in sequences:
        seq_dir = os.path.join(dataDir, folder, seq)
        if not os.path.isdir(seq_dir):
            print(f"Sequence folder not found for {seq} in {folder}: {seq_dir}")
            continue

        imagePath, maskPath = None, None

        # 遍历序列目录，匹配影像文件与对应掩膜命名
        try:
            for file in os.listdir(seq_dir):
                filePath = os.path.join(seq_dir, file)
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    if seq == 'DCE' and 'DCE' in file:
                        imagePath = filePath
                        maskPath = os.path.join(seq_dir, 'DCE_mask_resample.nii.gz')
                    elif seq == 'mappost' and 'mappost' in file:
                        imagePath = filePath
                        maskPath = os.path.join(seq_dir, 'mappost_mask_resample.nii.gz')
                    elif seq == 'mappre' and 'mappre' in file:
                        imagePath = filePath
                        maskPath = os.path.join(seq_dir, 'mappre_mask_resample.nii.gz')
                    elif seq == 'stir' and 'stir' in file:
                        imagePath = filePath
                        maskPath = os.path.join(seq_dir, 'stir_mask_resample.nii.gz')
                    elif seq == 'ADC' and 'ADC' in file:
                        imagePath = filePath
                        maskPath = os.path.join(seq_dir, 'ADC_mask_resample.nii.gz')
        except Exception as e:
            print(f"Error listing files in {seq_dir}: {e}")
            continue

        # 路径与文件存在性校验
        if not imagePath or not os.path.isfile(imagePath):
            print(f"Image file not found for {seq} in {folder}: {imagePath}")
            continue
        if not maskPath or not os.path.isfile(maskPath):
            print(f"Mask file not found for {seq} in {folder}: {maskPath}")
            continue

        # 读取影像与掩膜
        try:
            image = sitk.ReadImage(imagePath)
            mask = sitk.ReadImage(maskPath)
            print(f"Files loaded successfully for {seq} in {folder}: {imagePath}, {maskPath}")
        except Exception as e:
            print(f"Error reading files {imagePath} and {maskPath} for {seq}: {e}")
            continue

        # 若网格不一致则将掩膜重采样到影像网格（尺寸/间距/方向/原点任一不等）
        needs_resample = (
            image.GetSize() != mask.GetSize() or
            image.GetSpacing() != mask.GetSpacing() or
            image.GetDirection() != mask.GetDirection() or
            image.GetOrigin() != mask.GetOrigin()
        )
        if needs_resample:
            print(f"Resampling mask for {seq} in {folder} to match image grid")
            mask = sitk.Resample(
                mask,              # 源
                image,             # 参考（对齐到影像网格）
                sitk.Transform(),  # 恒等变换
                sitk.sitkNearestNeighbor,  # 最近邻插值，避免标签污染
                0.0,               # 默认填充值
                mask.GetPixelID()  # 保持掩膜像素类型
            )

        # ===== 关键改动：传入内存对象，确保用到重采样后的掩膜 =====
        try:
            featureVector = extractor.execute(image, mask)
        except Exception as e:
            print(f"PyRadiomics execute failed for {seq} in {folder}: {e}")
            continue

        # 转为单行 DataFrame，并添加病人与序列标识
        df_add = pd.DataFrame([featureVector])
        df_add.insert(0, 'PatientID', folder)
        df_add.insert(1, 'Sequence', seq)

        patient_features.append(df_add)

    # 汇总当前病人的特征
    if len(patient_features) > 0:
        patient_df = pd.concat(patient_features, ignore_index=True)
        df = pd.concat([df, patient_df], ignore_index=True)
    else:
        print(f"No features extracted for patient {folder}, skipping.")

# ========= 写出 Excel =========
out_path = os.path.join(dataDir, 'results_reader2.xlsx')  # 保持原文件名
df.to_excel(out_path, index=False)
print(f"Done. Saved features to: {out_path}")


# In[ ]:




