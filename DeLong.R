# ===================== Packages =====================
packages <- c("pROC", "readxl", "dplyr")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ===================== Helpers =====================
to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

fix_df <- function(df){
  # 统一列类型，去掉缺失
  df$PatientID <- as.character(df$PatientID)
  df$Label <- to_num(df$Label)
  df$RadiomicsScore <- to_num(df$RadiomicsScore)
  df <- df[complete.cases(df[, c("PatientID","Label","RadiomicsScore")]), ]
  return(df)
}

# 读取一个文件并改名为指定模型列
read_one_model <- function(path, model_key){
  df <- readxl::read_excel(path)
  need_cols <- c("PatientID", "RadiomicsScore", "Label")
  miss <- setdiff(need_cols, colnames(df))
  if (length(miss) > 0) {
    stop(paste0("文件缺少列: ", paste(miss, collapse = ", "), "\n文件: ", path))
  }
  df <- fix_df(df)
  df <- df %>%
    dplyr::select(PatientID, Label, RadiomicsScore) %>%
    dplyr::rename(!!model_key := RadiomicsScore)
  return(df)
}

# 按 PatientID + Label 对齐合并（保证 paired=TRUE 的前提）
load_and_align <- function(file_map){
  # file_map: 命名向量/列表，names=模型名，values=路径
  dfs <- list()
  for (nm in names(file_map)) {
    dfs[[nm]] <- read_one_model(file_map[[nm]], nm)
  }
  # 逐个 inner_join，确保完全对齐
  merged <- dfs[[1]]
  if (length(dfs) > 1) {
    for (i in 2:length(dfs)) {
      merged <- dplyr::inner_join(merged, dfs[[i]], by = c("PatientID","Label"))
    }
  }
  return(merged)
}

# 计算 ROC（在同一对齐数据上）
calc_rocs <- function(aligned_df, model_names){
  rocs <- list()
  for (m in model_names) {
    rocs[[m]] <- pROC::roc(aligned_df$Label, aligned_df[[m]], quiet = TRUE)
  }
  return(rocs)
}

# 输出 AUC
print_aucs <- function(rocs, cohort_name){
  cat("\n====================", cohort_name, "AUC ====================\n")
  for (m in names(rocs)) {
    cat(sprintf("AUC %-14s : %.4f\n", m, as.numeric(pROC::auc(rocs[[m]]))))
  }
}

# DeLong 两两比较（所有组合）
delong_pairwise <- function(rocs, cohort_name){
  ms <- names(rocs)
  pairs <- combn(ms, 2, simplify = FALSE)
  cat("\n====================", cohort_name, "DeLong (paired) ====================\n")
  out <- list()
  for (p in pairs) {
    m1 <- p[1]; m2 <- p[2]
    # pROC::roc.test 输出里包含 statistic / p.value / conf.int 等
    test_res <- pROC::roc.test(rocs[[m1]], rocs[[m2]], method = "delong", paired = TRUE)
    out_name <- paste0(m1, " vs ", m2)
    out[[out_name]] <- test_res
    cat("\n--- ", out_name, " ---\n", sep = "")
    print(test_res)
  }
  invisible(out)
}

# ===================== File paths =====================
train_files <- c(
  "Regular"       = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_regular.xlsx",
  "Regular + Map" = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_regular+map.xlsx",
  "Fusion"        = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_radiomics_scores_fusion.xlsx",
  "Clinical"      = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_train_clinical_scores.xlsx"
)

test_files <- c(
  "Regular"       = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular.xlsx",
  "Regular + Map" = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_regular+map.xlsx",
  "Fusion"        = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_radiomics_scores_fusion.xlsx",
  "Clinical"      = "C:/Users/Sun/Desktop/3dslicer_malignant_nii/LR/LR_test_clinical_scores.xlsx"
)

model_names <- names(train_files)

# ===================== Train =====================
cat("\n📌 Processing TRAIN...\n")
train_aligned <- load_and_align(train_files)
cat("✅ TRAIN aligned n = ", nrow(train_aligned),
    " | positives = ", sum(train_aligned$Label == 1, na.rm = TRUE), "\n", sep = "")

rocs_train <- calc_rocs(train_aligned, model_names)
print_aucs(rocs_train, "TRAIN")
delong_train <- delong_pairwise(rocs_train, "TRAIN")

# ===================== Test =====================
cat("\n📌 Processing TEST...\n")
test_aligned <- load_and_align(test_files)
cat("✅ TEST aligned n = ", nrow(test_aligned),
    " | positives = ", sum(test_aligned$Label == 1, na.rm = TRUE), "\n", sep = "")

rocs_test <- calc_rocs(test_aligned, model_names)
print_aucs(rocs_test, "TEST")
delong_test <- delong_pairwise(rocs_test, "TEST")

cat("\n✅ Done.\n")
