import pandas as pd

mappre_df = pd.read_excel(r"C:\Users\Sun\Desktop\3d_slicer_malignant_nii\mappre.xlsx")
mappost_df = pd.read_excel(r"C:\Users\Sun\Desktop\3d_slicer_malignant_nii\mappost.xlsx")

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

delta1_df.rename(columns={"delta1": "Sequence"}, inplace=True)
delta2_df.rename(columns={"delta2": "Sequence"}, inplace=True)

delta1_df.to_excel(r"C:\Users\Sun\Desktop\3d_slicer_malignant_nii\delta1.xlsx", index=False)
delta2_df.to_excel(r"C:\Users\Sun\Desktop\3d_slicer_malignant_nii\delta2.xlsx", index=False)

print("Delta1 DataFrame:")
print(delta1_df.head())

print("Delta2 DataFrame:")
print(delta2_df.head())







