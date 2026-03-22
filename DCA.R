# ===================== Packages =====================
packages <- c("readxl", "dplyr", "ggplot2", "dcurves", "scales")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ===================== Config =====================
model_list <- c("regular", "regular+map", "fusion", "clinical")

base_path <- "C:/Users/Sun/Desktop/3dslicer_malignant_nii"
save_fig_path <- file.path(base_path, "DCA")
dir.create(save_fig_path, showWarnings = FALSE, recursive = TRUE)

display_names <- c(
  "regular"     = "Regular",
  "regular+map" = "Regular + Map",
  "fusion"      = "Fusion",
  "clinical"    = "Clinical"
)

# ✅ 配色（按你给的配色 + Clinical 改成湖蓝更区分 Fusion）
color_map <- c(
  "Treat All"      = "#E51F21",
  "Treat None"     = "#528FC2",
  "Regular"        = "#63B960",
  "Regular + Map"  = "#9E59A8",
  "Fusion"         = "#FF7B06",
  "Clinical"       = "#00BFC4"
)

# 线型：全部实线（如需区分可把 Clinical 改成 "dotdash"）
linetype_map <- c(
  "Treat All"      = "solid",
  "Treat None"     = "solid",
  "Regular"        = "solid",
  "Regular + Map"  = "solid",
  "Fusion"         = "solid",
  "Clinical"       = "solid"
)

# 线宽
linewidth_map <- c(
  "Treat All"      = 1.05,
  "Treat None"     = 1.05,
  "Regular"        = 1.05,
  "Regular + Map"  = 1.05,
  "Fusion"         = 1.05,
  "Clinical"       = 1.05
)

# 平滑参数（越大越平滑，建议 0.25~0.4 之间微调）
smooth_span <- 0.25

# 每个数据集单独设置 x 最大值：test=0.8, train=1.0
x_max_map <- list(test = 0.8, train = 1.0)

# ✅【关键：固定 y 轴范围 + 固定 y 轴刻度/格式】——用于 PPT 对齐
y_lim_fixed    <- c(-0.12, 0.30)
y_breaks_fixed <- seq(-0.10, 0.30, 0.10)

# ===================== Load & Merge =====================
load_combined_data <- function(dataset_type) {
  full_data <- NULL
  
  for (model in model_list) {
    
    if (model == "clinical") {
      file_path <- file.path(base_path, "LR", paste0("LR_", dataset_type, "_clinical_scores.xlsx"))
    } else {
      file_path <- file.path(base_path, "LR", paste0("LR_", dataset_type, "_radiomics_scores_", model, ".xlsx"))
    }
    
    if (!file.exists(file_path)) {
      cat("❌ 文件不存在:", file_path, "\n")
      next
    }
    
    df_raw <- readxl::read_excel(file_path)
    
    need_cols <- c("PatientID", "RadiomicsScore", "Label")
    miss <- setdiff(need_cols, colnames(df_raw))
    if (length(miss) > 0) {
      cat("❌ 文件缺少列:", paste(miss, collapse = ", "), "\n")
      cat("   文件:", file_path, "\n")
      cat("   现有列:", paste(colnames(df_raw), collapse = ", "), "\n")
      next
    }
    
    df <- df_raw %>%
      dplyr::select(PatientID, RadiomicsScore, Label) %>%
      dplyr::mutate(
        PatientID = as.character(PatientID),
        Label = as.integer(Label)
      ) %>%
      dplyr::rename(!!paste0("pred_", model) := RadiomicsScore)
    
    if (is.null(full_data)) {
      full_data <- df
    } else {
      n_before <- nrow(full_data)
      full_data <- dplyr::inner_join(full_data, df, by = c("PatientID", "Label"))
      n_after <- nrow(full_data)
      if (n_after < n_before) {
        cat("⚠️ [行数减少] inner_join:", n_before, "->", n_after, "（模型:", model, "）\n")
      } else {
        cat("✅ [合并成功] inner_join:", n_before, "->", n_after, "（模型:", model, "）\n")
      }
    }
  }
  
  if (!is.null(full_data) && nrow(full_data) > 0) {
    cat("📌 合并完成：", dataset_type, "n=", nrow(full_data),
        "阳性=", sum(full_data$Label == 1, na.rm = TRUE), "\n")
  }
  return(full_data)
}

# ===================== Helper: LOESS smooth per curve =====================
smooth_curve <- function(df, span = 0.30) {
  df <- df[order(df$threshold), ]
  if (nrow(df) < 8) {
    df$net_benefit_s <- df$net_benefit
    return(df)
  }
  fit <- try(stats::loess(net_benefit ~ threshold, data = df, span = span, degree = 1), silent = TRUE)
  if (inherits(fit, "try-error")) {
    df$net_benefit_s <- df$net_benefit
  } else {
    df$net_benefit_s <- stats::predict(fit, newdata = df$threshold)
  }
  df
}

# ===================== Plot DCA (custom ggplot, single legend) =====================
plot_combined_dca <- function(data, dataset_type, save_path) {
  if (is.null(data) || nrow(data) == 0) {
    cat("⚠️ 没有可用数据用于绘图\n")
    return(NULL)
  }
  
  required_cols <- paste0("pred_", model_list)
  missing_cols <- setdiff(required_cols, colnames(data))
  if (length(missing_cols) > 0) {
    cat("⚠️ 缺少列:", paste(missing_cols, collapse = ", "), "\n")
    return(NULL)
  }
  
  x_max <- x_max_map[[dataset_type]]
  thr <- seq(0.01, x_max - 0.01, 0.01)
  
  formula_str <- paste("Label ~", paste0("`", paste0("pred_", model_list), "`", collapse = " + "))
  dca_formula <- as.formula(formula_str)
  
  label_map <- setNames(display_names[model_list], paste0("pred_", model_list))
  label_map <- as.list(label_map)
  
  legend_levels <- c("Treat All", "Treat None", display_names[model_list])
  
  tryCatch({
    dca_result <- dca(
      formula = dca_formula,
      data = data,
      thresholds = thr,
      label = label_map
    )
    
    d <- dca_result$dca
    d$label <- factor(d$label, levels = legend_levels)
    
    # 平滑：只对模型曲线做 loess
    d_models <- d %>%
      dplyr::filter(label %in% display_names[model_list]) %>%
      dplyr::group_by(label) %>%
      dplyr::group_modify(~smooth_curve(.x, span = smooth_span)) %>%
      dplyr::ungroup()
    
    d_base <- d %>% dplyr::filter(label %in% c("Treat All", "Treat None"))
    d_base$net_benefit_s <- d_base$net_benefit
    
    d_plot <- dplyr::bind_rows(d_base, d_models)
    
    dataset_title <- ifelse(dataset_type == "train", "训练集", "测试集")
    file_tag <- ifelse(dataset_type == "train", "4models_thr0_100", "4models_thr0_80")
    
    p <- ggplot(
      d_plot,
      aes(
        x = threshold,
        y = net_benefit_s,
        color = label,
        linetype = label,
        linewidth = label
      )
    ) +
      geom_line() +
      labs(
        title = paste0(dataset_title, " - 决策曲线分析"),
        x = "High Risk Threshold",
        y = "Net Benefit",
        color = "模型"
      ) +
      scale_x_continuous(
        limits = c(0, x_max),
        breaks = seq(0, x_max, 0.2),
        labels = scales::percent_format(accuracy = 1)
      ) +
      # ✅ 固定 y 轴刻度/格式（两位小数，保证左右留白一致）
      scale_y_continuous(
        breaks = y_breaks_fixed,
        labels = function(x) sprintf("%.2f", x)
      ) +
      # ✅ 固定 y 轴范围（保证 train/test panel 结构一致）
      coord_cartesian(ylim = y_lim_fixed) +
      scale_color_manual(values = color_map, breaks = legend_levels) +
      scale_linetype_manual(values = linetype_map, breaks = legend_levels) +
      scale_linewidth_manual(values = linewidth_map, breaks = legend_levels) +
      guides(
        linetype = "none",
        linewidth = "none",
        color = guide_legend(
          title = "模型",
          override.aes = list(
            linetype = unname(linetype_map[legend_levels]),
            linewidth = unname(linewidth_map[legend_levels])
          )
        )
      ) +
      theme_minimal(base_size = 16) +
      theme(
        panel.grid = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        legend.position = "right",
        legend.text  = element_text(size = 13),
        legend.title = element_text(size = 14, face = "bold"),
        axis.title   = element_text(size = 16, face = "bold"),
        axis.text    = element_text(size = 13),
        plot.title   = element_text(size = 18, face = "bold", hjust = 0.5),
        # ✅ 固定外边距（进一步减少 PPT 里“看起来不齐”的概率）
        plot.margin = margin(12, 18, 10, 16)  # 上右下左，可微调
      )
    
    out_png <- file.path(save_path, paste0(dataset_type, "_Combined_DCA_", file_tag, "_smooththin.png"))
    out_pdf <- file.path(save_path, paste0(dataset_type, "_Combined_DCA_", file_tag, "_smooththin.pdf"))
    
    ggsave(out_png, plot = p, width = 10, height = 8, dpi = 300, bg = "white")
    ggsave(out_pdf, plot = p, width = 10, height = 8, bg = "white")
    
    cat("✅ 保存完成:", out_png, "\n")
    cat("✅ 保存完成:", out_pdf, "\n")
    return(p)
    
  }, error = function(e) {
    cat("❌ DCA 分析失败:", e$message, "\n")
    return(NULL)
  })
}

# ===================== Run =====================
for (dataset_type in c("train", "test")) {
  cat("\n📌 正在处理：", dataset_type, "\n")
  combined_data <- load_combined_data(dataset_type)
  if (!is.null(combined_data) && nrow(combined_data) > 0) {
    plot_combined_dca(combined_data, dataset_type, save_fig_path)
  } else {
    cat("⚠️ 没有可用数据用于绘图\n")
  }
}

cat("\n📊 DCA 完成！输出目录：", save_fig_path, "\n")
