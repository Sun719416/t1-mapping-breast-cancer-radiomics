#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.executable)


import radiomics
from radiomics import featureextractor
print(radiomics.__version__)


import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

dataDir = r"C:\Users\Sun\Desktop\resampled2"
sequences = ['stir', 'mappre', 'mappost', 'DCE', 'ADC']

folderList = [f for f in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, f))]

extractor = featureextractor.RadiomicsFeatureExtractor()  

extractor.enableImageNormalization = True
extractor.imageNormalizingMethod = "z-score"

extractor.enableImageTypes(Wavelet={})

df = pd.DataFrame()

for folder in folderList:
    patient_features = []

    for seq in sequences:
        seq_dir = os.path.join(dataDir, folder, seq)
        if not os.path.isdir(seq_dir):
            print(f"Sequence folder not found for {seq} in {folder}: {seq_dir}")
            continue

        imagePath, maskPath = None, None

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

        if not imagePath or not os.path.isfile(imagePath):
            print(f"Image file not found for {seq} in {folder}: {imagePath}")
            continue
        if not maskPath or not os.path.isfile(maskPath):
            print(f"Mask file not found for {seq} in {folder}: {maskPath}")
            continue

        try:
            image = sitk.ReadImage(imagePath)
            mask = sitk.ReadImage(maskPath)
            print(f"Files loaded successfully for {seq} in {folder}: {imagePath}, {maskPath}")
        except Exception as e:
            print(f"Error reading files {imagePath} and {maskPath} for {seq}: {e}")
            continue

        needs_resample = (
            image.GetSize() != mask.GetSize() or
            image.GetSpacing() != mask.GetSpacing() or
            image.GetDirection() != mask.GetDirection() or
            image.GetOrigin() != mask.GetOrigin()
        )
        if needs_resample:
            print(f"Resampling mask for {seq} in {folder} to match image grid")
            mask = sitk.Resample(
                mask,              
                image,          
                sitk.Transform(), 
                sitk.sitkNearestNeighbor,  
                0.0,             
                mask.GetPixelID()  
            )

        try:
            featureVector = extractor.execute(image, mask)
        except Exception as e:
            print(f"PyRadiomics execute failed for {seq} in {folder}: {e}")
            continue

        df_add = pd.DataFrame([featureVector])
        df_add.insert(0, 'PatientID', folder)
        df_add.insert(1, 'Sequence', seq)

        patient_features.append(df_add)
    if len(patient_features) > 0:
        patient_df = pd.concat(patient_features, ignore_index=True)
        df = pd.concat([df, patient_df], ignore_index=True)
    else:
        print(f"No features extracted for patient {folder}, skipping.")

out_path = os.path.join(dataDir, 'results_reader2.xlsx')  
df.to_excel(out_path, index=False)
print(f"Done. Saved features to: {out_path}")





