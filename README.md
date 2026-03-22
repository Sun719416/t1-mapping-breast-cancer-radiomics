# t1-mapping-breast-cancer-radiomics
## Code Description

- `T1 feature_histogram_features.py`: extracts histogram features from the original grayscale T1 mapping images.
- `Radiomics_featureextractor.py`: extracts radiomic features from weighted MR images.
- `feature_selection.py`: performs feature selection for each dataset.
- `model_construction.py`: performs model construction for each dataset.

Feature selection and model construction were performed separately for two datasets:
1. the conventional MRI feature set
2. the conventional MRI plus T1-mapping feature set
