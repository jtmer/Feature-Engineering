import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

DEFAULT_ROOT = "."
DEFAULT_GLOB = "results/results_monthly_backtest_*"

# 目录名中参数别名 -> 完整名
HP_ALIAS_TO_NAME = {
    "lls": "lambda_latent_scale",
    "lrt": "latent_rms_target",
    "tlr": "train_lr",
    "lpr": "lambda_past_recon",
    "lfp": "lambda_future_pred",
    "lsp": "lambda_stats_pred",
    "lyps": "lambda_y_patch_std",
    "lpf": "lambda_proxy_floor",
    "psf": "proxy_sens_floor",
    "lrf": "lambda_ratio_floor",
    "pcf": "proxy_cov_ratio_floor",
    "lmon": "lambda_monitor",
    "lsf": "latent_std_floor",
    "llsf": "lambda_latent_std_floor",
    "dlr": "decoder_lr",
}

HP_NAME_ORDER = [
    "lambda_latent_scale",
    "latent_rms_target",
    "train_lr",
    "lambda_past_recon",
    "lambda_future_pred",
    "lambda_stats_pred",
    "lambda_y_patch_std",
    "lambda_proxy_floor",
    "proxy_sens_floor",
    "lambda_ratio_floor",
    "proxy_cov_ratio_floor",
    "lambda_monitor",
    "latent_std_floor",
    "lambda_latent_std_floor",
    "decoder_lr",
]

MAIN_METRICS = ["iMAE", "iMSE"]


# ============================================================
# Utils
# ============================================================

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def suffix_token_to_float(s: str):
    """
    将目录后缀中的数值字符串转成 float。
    例如：
      0p0001 -> 0.0001
      5em05  -> 5e-05
      1      -> 1.0
      0p4    -> 0.4
    """
    if s is None or s == "":
        return np.nan

    s = str(s).strip()

    # 先处理 em / e
    # 例如 5em05 -> 5e-05
    if "em" in s:
        s = s.replace("em", "e-")
    # 普通 p -> .
    s = s.replace("p", ".")

    try:
        return float(s)
    except Exception:
        return np.nan


def format_metric(v):
    if pd.isna(v):
        return "nan"
    if abs(v) >= 1000:
        return f"{v:,.4f}"
    return f"{v:.6f}"


def robust_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return df.iloc[:, 0].corr(df.iloc[:, 1], method=method)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """
    两个指标都越小越好。
    返回一个 bool Series，表示是否在 Pareto front 上。
    """
    mask_valid = df[[x_col, y_col]].notna().all(axis=1)
    sub = df.loc[mask_valid, [x_col, y_col]].copy()
    idxs = sub.index.tolist()
    vals = sub[[x_col, y_col]].values

    pareto = []
    for i in range(len(vals)):
        xi, yi = vals[i]
        dominated = False
        for j in range(len(vals)):
            if i == j:
                continue
            xj, yj = vals[j]
            # j dominates i
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
        pareto.append(not dominated)

    out = pd.Series(False, index=df.index)
    out.loc[idxs] = pareto
    return out


def normalize_rank_score(s: pd.Series, ascending: bool = True) -> pd.Series:
    """
    rank-based score in [0,1], 越大越好。
    ascending=True 表示值越小越好。
    """
    v = s.copy()
    if ascending:
        r = v.rank(method="average", ascending=True)
    else:
        r = v.rank(method="average", ascending=False)

    n = r.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=s.index)
    return 1.0 - (r - 1.0) / (n - 1.0)


# ============================================================
# Parse experiment suffix
# ============================================================

def parse_exp_suffix(exp_name: str) -> Dict[str, float]:
    """
    输入：
      results_monthly_backtest_lls0p0001_lrt6_tlr5em05_...
    输出：
      {
        "lambda_latent_scale": 0.0001,
        "latent_rms_target": 6.0,
        ...
      }
    """
    prefix = "results_monthly_backtest_"
    suffix = exp_name
    if suffix.startswith(prefix):
        suffix = suffix[len(prefix):]

    res = {k: np.nan for k in HP_NAME_ORDER}

    # 用 alias 前缀匹配每个 token
    tokens = suffix.split("_")
    for token in tokens:
        matched = False
        for alias, full_name in HP_ALIAS_TO_NAME.items():
            if token.startswith(alias):
                num_str = token[len(alias):]
                res[full_name] = suffix_token_to_float(num_str)
                matched = True
                break
        if not matched:
            # 忽略不认识的 token
            pass

    return res


# ============================================================
# Read each experiment
# ============================================================

def read_one_experiment(exp_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    summary_csv = exp_dir / "results_summary.csv"
    if not summary_csv.exists():
        return None, None

    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        print(f"[skip] read failed: {summary_csv} | err={e}", flush=True)
        return None, None

    required_cols = {"month", "iMSE", "iMAE"}
    if not required_cols.issubset(df.columns):
        print(f"[skip] missing required cols in {summary_csv}", flush=True)
        return None, None

    # 只保留我们关心的列，其他列如果有也不影响
    keep_cols = [c for c in df.columns if c in {"month", "iMSE", "iMAE", "iMAPE", "iMAPE_ad", "coverage", "n_windows", "n_days_used"}]
    df = df[keep_cols].copy()
    df["month"] = df["month"].astype(str)

    hp = parse_exp_suffix(exp_dir.name)

    rec = {
        "exp_name": exp_dir.name,
        "exp_dir": str(exp_dir),
        "n_months": len(df),
        "months": ",".join(df["month"].tolist()),
    }
    rec.update(hp)

    # mean
    rec["mean_iMAE"] = df["iMAE"].mean()
    rec["mean_iMSE"] = df["iMSE"].mean()

    # median
    rec["median_iMAE"] = df["iMAE"].median()
    rec["median_iMSE"] = df["iMSE"].median()

    # std
    rec["std_iMAE"] = df["iMAE"].std(ddof=0)
    rec["std_iMSE"] = df["iMSE"].std(ddof=0)

    # worst / best
    rec["worst_iMAE"] = df["iMAE"].max()
    rec["worst_iMSE"] = df["iMSE"].max()
    rec["best_iMAE"] = df["iMAE"].min()
    rec["best_iMSE"] = df["iMSE"].min()

    # worst month
    rec["worst_month_iMAE"] = df.loc[df["iMAE"].idxmax(), "month"]
    rec["worst_month_iMSE"] = df.loc[df["iMSE"].idxmax(), "month"]

    # stability (越小越好)
    rec["cv_iMAE"] = rec["std_iMAE"] / (rec["mean_iMAE"] + 1e-12)
    rec["cv_iMSE"] = rec["std_iMSE"] / (rec["mean_iMSE"] + 1e-12)

    rec["worst_mean_ratio_iMAE"] = rec["worst_iMAE"] / (rec["mean_iMAE"] + 1e-12)
    rec["worst_mean_ratio_iMSE"] = rec["worst_iMSE"] / (rec["mean_iMSE"] + 1e-12)

    # 稳定性综合分数（越小越好）
    rec["stability_score"] = 0.5 * rec["cv_iMAE"] + 0.5 * rec["worst_mean_ratio_iMAE"]

    # 综合性能分数（只看 iMAE + iMSE，越小越好）
    rec["raw_joint_score"] = 0.5 * rec["mean_iMAE"] + 0.5 * math.sqrt(max(rec["mean_iMSE"], 0.0))

    return df, rec


def collect_all_experiments(root: Path, pattern: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    summary_rows = []

    exp_dirs = sorted([p for p in root.glob(pattern) if p.is_dir()])

    for exp_dir in exp_dirs:
        df_month, rec = read_one_experiment(exp_dir)
        if df_month is None or rec is None:
            continue

        summary_rows.append(rec)

        df_tmp = df_month.copy()
        df_tmp["exp_name"] = exp_dir.name
        df_tmp["exp_dir"] = str(exp_dir)

        hp = parse_exp_suffix(exp_dir.name)
        for k, v in hp.items():
            df_tmp[k] = v

        monthly_rows.append(df_tmp)

    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.concat(monthly_rows, axis=0, ignore_index=True) if monthly_rows else pd.DataFrame()

    return summary_df, monthly_df


# ============================================================
# Analysis
# ============================================================

def add_rank_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    df["rank_score_iMAE"] = normalize_rank_score(df["mean_iMAE"], ascending=True)
    df["rank_score_iMSE"] = normalize_rank_score(df["mean_iMSE"], ascending=True)
    df["rank_score_stability"] = normalize_rank_score(df["stability_score"], ascending=True)

    # 你要求不看 coverage，这里综合分数只基于 iMAE / iMSE / 稳定性
    df["overall_score"] = (
        0.45 * df["rank_score_iMAE"] +
        0.35 * df["rank_score_iMSE"] +
        0.20 * df["rank_score_stability"]
    )

    return df


def analyze_hyperparam_vs_performance(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    每个超参数：
      - 与 mean_iMAE / mean_iMSE / stability_score 的 spearman 相关
      - 分组均值范围（近似看敏感度）
    """
    rows = []

    for hp in HP_NAME_ORDER:
        if hp not in summary_df.columns:
            continue

        s = summary_df[hp]
        if s.dropna().nunique() <= 1:
            continue

        rec = {
            "hyperparam": hp,
            "n_unique": int(s.dropna().nunique()),
            "spearman_mean_iMAE": robust_corr(s, summary_df["mean_iMAE"], method="spearman"),
            "spearman_mean_iMSE": robust_corr(s, summary_df["mean_iMSE"], method="spearman"),
            "spearman_stability_score": robust_corr(s, summary_df["stability_score"], method="spearman"),
        }

        # 分组均值范围：越大说明这个参数切换取值时，性能均值变化越大
        grp = summary_df.groupby(hp, dropna=True).agg(
            mean_iMAE=("mean_iMAE", "mean"),
            mean_iMSE=("mean_iMSE", "mean"),
            stability_score=("stability_score", "mean"),
            count=("exp_name", "count"),
        ).reset_index()

        if len(grp) >= 2:
            rec["group_mean_iMAE_range"] = grp["mean_iMAE"].max() - grp["mean_iMAE"].min()
            rec["group_mean_iMSE_range"] = grp["mean_iMSE"].max() - grp["mean_iMSE"].min()
            rec["group_stability_range"] = grp["stability_score"].max() - grp["stability_score"].min()

            # 最优取值（按 group mean）
            rec["best_value_by_mean_iMAE"] = grp.loc[grp["mean_iMAE"].idxmin(), hp]
            rec["best_value_by_mean_iMSE"] = grp.loc[grp["mean_iMSE"].idxmin(), hp]
            rec["best_value_by_stability"] = grp.loc[grp["stability_score"].idxmin(), hp]
        else:
            rec["group_mean_iMAE_range"] = np.nan
            rec["group_mean_iMSE_range"] = np.nan
            rec["group_stability_range"] = np.nan
            rec["best_value_by_mean_iMAE"] = np.nan
            rec["best_value_by_mean_iMSE"] = np.nan
            rec["best_value_by_stability"] = np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # 重要性排序分数：综合相关绝对值 + 分组均值变化范围
    # 为避免量纲差异，用 rank score 融合
    out["rank_abs_corr_iMAE"] = normalize_rank_score(out["spearman_mean_iMAE"].abs(), ascending=False)
    out["rank_abs_corr_iMSE"] = normalize_rank_score(out["spearman_mean_iMSE"].abs(), ascending=False)
    out["rank_range_iMAE"] = normalize_rank_score(out["group_mean_iMAE_range"], ascending=False)
    out["rank_range_iMSE"] = normalize_rank_score(out["group_mean_iMSE_range"], ascending=False)

    out["importance_score"] = (
        0.35 * out["rank_abs_corr_iMAE"] +
        0.25 * out["rank_abs_corr_iMSE"] +
        0.25 * out["rank_range_iMAE"] +
        0.15 * out["rank_range_iMSE"]
    )

    out = out.sort_values("importance_score", ascending=False).reset_index(drop=True)
    return out


def detect_unstable_or_overfit_like_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    不看 train/val gap，因此这里的“过拟合模型”定义为：
      - 平均误差不算差，但跨月波动大 / 最差月明显崩
    """
    df = summary_df.copy()

    # 低 mean 好、高 stability_score 差
    mae_good_q = df["mean_iMAE"].quantile(0.35)
    mse_good_q = df["mean_iMSE"].quantile(0.35)
    unstable_q = df["stability_score"].quantile(0.75)
    ratio_q = df["worst_mean_ratio_iMAE"].quantile(0.75)

    cond = (
        (
            (df["mean_iMAE"] <= mae_good_q) |
            (df["mean_iMSE"] <= mse_good_q)
        ) &
        (
            (df["stability_score"] >= unstable_q) |
            (df["worst_mean_ratio_iMAE"] >= ratio_q)
        )
    )

    out = df.loc[cond].copy()
    out["overfit_like_score"] = (
        normalize_rank_score(out["mean_iMAE"], ascending=True).fillna(0.0) * 0.4 +
        normalize_rank_score(out["stability_score"], ascending=False).fillna(0.0) * 0.6
    )
    out = out.sort_values(
        ["stability_score", "worst_mean_ratio_iMAE", "mean_iMAE"],
        ascending=[False, False, True]
    )
    return out


def suggest_top3_hyperparams(importance_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    给出最值得继续细调的 3 个参数：
      1) 重要性高
      2) 取值数 >= 2
      3) 最优值不在边界时更适合细调
    """
    if importance_df.empty:
        return importance_df

    rows = []
    for _, r in importance_df.iterrows():
        hp = r["hyperparam"]
        vals = sorted(summary_df[hp].dropna().unique().tolist())
        if len(vals) < 2:
            continue

        # 按 iMAE group mean 找当前最佳值
        grp = summary_df.groupby(hp, dropna=True)["mean_iMAE"].mean().reset_index()
        grp = grp.sort_values(hp).reset_index(drop=True)
        best_idx = grp["mean_iMAE"].idxmin()
        best_val = grp.loc[best_idx, hp]

        # 判断最佳值是不是边界
        is_left_edge = math.isclose(best_val, grp[hp].min(), rel_tol=1e-12, abs_tol=1e-12)
        is_right_edge = math.isclose(best_val, grp[hp].max(), rel_tol=1e-12, abs_tol=1e-12)
        on_edge = is_left_edge or is_right_edge

        # 推荐动作
        if on_edge and is_left_edge:
            suggestion = "best_at_left_edge -> 建议往更小方向补点"
        elif on_edge and is_right_edge:
            suggestion = "best_at_right_edge -> 建议往更大方向补点"
        else:
            suggestion = "best_in_middle -> 建议在最佳值附近做更细网格"

        rows.append({
            "hyperparam": hp,
            "importance_score": r["importance_score"],
            "n_unique": r["n_unique"],
            "best_value_by_mean_iMAE": r["best_value_by_mean_iMAE"],
            "best_value_by_mean_iMSE": r["best_value_by_mean_iMSE"],
            "group_mean_iMAE_range": r["group_mean_iMAE_range"],
            "group_mean_iMSE_range": r["group_mean_iMSE_range"],
            "spearman_mean_iMAE": r["spearman_mean_iMAE"],
            "spearman_mean_iMSE": r["spearman_mean_iMSE"],
            "best_location": "edge" if on_edge else "middle",
            "next_action": suggestion,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 更适合细调的优先级：
    # importance 高，且最佳值在中间更优先
    out["middle_bonus"] = (out["best_location"] == "middle").astype(float)
    out["refine_priority"] = out["importance_score"] + 0.08 * out["middle_bonus"]

    out = out.sort_values("refine_priority", ascending=False).reset_index(drop=True)
    return out.head(3)


# ============================================================
# Plotting
# ============================================================

def save_corr_heatmap(summary_df: pd.DataFrame, out_dir: Path):
    cols = HP_NAME_ORDER + [
        "mean_iMAE", "mean_iMSE",
        "std_iMAE", "std_iMSE",
        "stability_score",
        "worst_mean_ratio_iMAE", "worst_mean_ratio_iMSE",
    ]
    cols = [c for c in cols if c in summary_df.columns]

    df = summary_df[cols].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    corr = df.corr(method="spearman")

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Spearman Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_spearman.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    corr.to_csv(out_dir / "heatmap_spearman.csv")


def save_pareto_plot(summary_df: pd.DataFrame, out_dir: Path):
    df = summary_df.copy()
    df["is_pareto"] = compute_pareto_front(df, "mean_iMAE", "mean_iMSE")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    non_p = df[~df["is_pareto"]]
    p = df[df["is_pareto"]]

    ax.scatter(non_p["mean_iMAE"], non_p["mean_iMSE"], alpha=0.55, label="non-pareto")
    ax.scatter(p["mean_iMAE"], p["mean_iMSE"], alpha=0.95, label="pareto-front")

    # 连接 pareto 前沿
    if len(p) > 0:
        p2 = p.sort_values("mean_iMAE")
        ax.plot(p2["mean_iMAE"], p2["mean_iMSE"], alpha=0.8)

    ax.set_xlabel("mean_iMAE (lower better)")
    ax.set_ylabel("mean_iMSE (lower better)")
    ax.set_title("Pareto Front: mean_iMAE vs mean_iMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_front_iMAE_iMSE.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    df.sort_values(["is_pareto", "mean_iMAE", "mean_iMSE"], ascending=[False, True, True]) \
      .to_csv(out_dir / "pareto_front_table.csv", index=False)


def save_hyperparam_vs_perf_plots(summary_df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "hyperparam_vs_perf"
    mkdir(plot_dir)

    for hp in HP_NAME_ORDER:
        if hp not in summary_df.columns:
            continue
        s = summary_df[hp]
        if s.dropna().nunique() <= 1:
            continue

        grp = summary_df.groupby(hp, dropna=True).agg(
            mean_iMAE=("mean_iMAE", "mean"),
            mean_iMSE=("mean_iMSE", "mean"),
            std_iMAE=("mean_iMAE", "std"),
            std_iMSE=("mean_iMSE", "std"),
            count=("exp_name", "count"),
        ).reset_index().sort_values(hp)

        # iMAE
        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_subplot(111)
        ax.plot(grp[hp], grp["mean_iMAE"], marker="o")
        ax.set_xlabel(hp)
        ax.set_ylabel("group mean_iMAE")
        ax.set_title(f"{hp} vs mean_iMAE")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{hp}_vs_mean_iMAE.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # iMSE
        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_subplot(111)
        ax.plot(grp[hp], grp["mean_iMSE"], marker="o")
        ax.set_xlabel(hp)
        ax.set_ylabel("group mean_iMSE")
        ax.set_title(f"{hp} vs mean_iMSE")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{hp}_vs_mean_iMSE.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        grp.to_csv(plot_dir / f"{hp}_group_summary.csv", index=False)


def save_monthly_best_heatmap(monthly_df: pd.DataFrame, out_dir: Path):
    """
    看每个月哪些实验/参数更稳。
    这里做实验 x 月份 很大，不直接画全图。
    改为：每个月 top1/top3 的参数分布输出表。
    """
    if monthly_df.empty:
        return

    rows = []
    for month, sub in monthly_df.groupby("month"):
        sub_mae = sub.sort_values("iMAE").head(3)
        sub_mse = sub.sort_values("iMSE").head(3)

        rec = {
            "month": month,
            "best_exp_by_iMAE": sub.sort_values("iMAE").iloc[0]["exp_name"],
            "best_iMAE": sub["iMAE"].min(),
            "best_exp_by_iMSE": sub.sort_values("iMSE").iloc[0]["exp_name"],
            "best_iMSE": sub["iMSE"].min(),
        }

        for i, (_, r) in enumerate(sub_mae.iterrows(), 1):
            rec[f"top{i}_exp_iMAE"] = r["exp_name"]
            rec[f"top{i}_iMAE"] = r["iMAE"]

        for i, (_, r) in enumerate(sub_mse.iterrows(), 1):
            rec[f"top{i}_exp_iMSE"] = r["exp_name"]
            rec[f"top{i}_iMSE"] = r["iMSE"]

        rows.append(rec)

    pd.DataFrame(rows).sort_values("month").to_csv(out_dir / "monthly_top_models.csv", index=False)


# ============================================================
# Text report
# ============================================================

def save_text_report(
    out_dir: Path,
    summary_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    overfit_like_df: pd.DataFrame,
    top3_refine_df: pd.DataFrame,
):
    report_file = out_dir / "analysis_report.txt"

    top_by_mae = summary_df.sort_values("mean_iMAE").head(10)
    top_by_mse = summary_df.sort_values("mean_iMSE").head(10)
    top_overall = summary_df.sort_values("overall_score", ascending=False).head(10)
    pareto_df = summary_df[summary_df["is_pareto"]].sort_values(["mean_iMAE", "mean_iMSE"])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("===== Analysis Report =====\n\n")
        f.write(f"n_experiments = {len(summary_df)}\n\n")

        f.write("----- Top 10 by mean_iMAE -----\n")
        for _, r in top_by_mae.iterrows():
            f.write(
                f"{r['exp_name']} | mean_iMAE={format_metric(r['mean_iMAE'])} | "
                f"mean_iMSE={format_metric(r['mean_iMSE'])} | "
                f"stability_score={format_metric(r['stability_score'])}\n"
            )

        f.write("\n----- Top 10 by mean_iMSE -----\n")
        for _, r in top_by_mse.iterrows():
            f.write(
                f"{r['exp_name']} | mean_iMSE={format_metric(r['mean_iMSE'])} | "
                f"mean_iMAE={format_metric(r['mean_iMAE'])} | "
                f"stability_score={format_metric(r['stability_score'])}\n"
            )

        f.write("\n----- Top 10 by overall_score -----\n")
        for _, r in top_overall.iterrows():
            f.write(
                f"{r['exp_name']} | overall_score={format_metric(r['overall_score'])} | "
                f"mean_iMAE={format_metric(r['mean_iMAE'])} | "
                f"mean_iMSE={format_metric(r['mean_iMSE'])} | "
                f"stability_score={format_metric(r['stability_score'])}\n"
            )

        f.write("\n----- Pareto Front (mean_iMAE vs mean_iMSE) -----\n")
        for _, r in pareto_df.iterrows():
            f.write(
                f"{r['exp_name']} | mean_iMAE={format_metric(r['mean_iMAE'])} | "
                f"mean_iMSE={format_metric(r['mean_iMSE'])}\n"
            )

        f.write("\n----- Overfit-like / unstable models -----\n")
        if len(overfit_like_df) == 0:
            f.write("None detected under current heuristic.\n")
        else:
            for _, r in overfit_like_df.head(20).iterrows():
                f.write(
                    f"{r['exp_name']} | mean_iMAE={format_metric(r['mean_iMAE'])} | "
                    f"worst_iMAE={format_metric(r['worst_iMAE'])} | "
                    f"worst_mean_ratio_iMAE={format_metric(r['worst_mean_ratio_iMAE'])} | "
                    f"stability_score={format_metric(r['stability_score'])}\n"
                )

        f.write("\n----- Hyperparameter importance -----\n")
        if len(importance_df) == 0:
            f.write("No valid hyperparameter importance results.\n")
        else:
            for _, r in importance_df.head(10).iterrows():
                f.write(
                    f"{r['hyperparam']} | importance_score={format_metric(r['importance_score'])} | "
                    f"best_value_by_mean_iMAE={r['best_value_by_mean_iMAE']} | "
                    f"best_value_by_mean_iMSE={r['best_value_by_mean_iMSE']}\n"
                )

        f.write("\n----- Top 3 hyperparameters to refine next -----\n")
        if len(top3_refine_df) == 0:
            f.write("No suggestion.\n")
        else:
            for _, r in top3_refine_df.iterrows():
                f.write(
                    f"{r['hyperparam']} | refine_priority={format_metric(r['refine_priority'])} | "
                    f"best_value_by_mean_iMAE={r['best_value_by_mean_iMAE']} | "
                    f"best_location={r['best_location']} | "
                    f"next_action={r['next_action']}\n"
                )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--glob", type=str, default=DEFAULT_GLOB)
    parser.add_argument("--out_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    mkdir(out_dir)

    print(f"[info] scanning root={root.resolve()} glob={args.glob}", flush=True)

    summary_df, monthly_df = collect_all_experiments(root, args.glob)

    if len(summary_df) == 0:
        print("[warn] no valid experiment found.", flush=True)
        return

    print(f"[info] loaded experiments = {len(summary_df)}", flush=True)

    summary_df = add_rank_scores(summary_df)
    summary_df["is_pareto"] = compute_pareto_front(summary_df, "mean_iMAE", "mean_iMSE")

    importance_df = analyze_hyperparam_vs_performance(summary_df)
    overfit_like_df = detect_unstable_or_overfit_like_models(summary_df)
    top3_refine_df = suggest_top3_hyperparams(importance_df, summary_df)

    # 保存基础表
    summary_df.sort_values("overall_score", ascending=False).to_csv(out_dir / "all_experiment_summary.csv", index=False)
    monthly_df.to_csv(out_dir / "all_monthly_results.csv", index=False)

    summary_df.sort_values("mean_iMAE").to_csv(out_dir / "rank_by_mean_iMAE.csv", index=False)
    summary_df.sort_values("mean_iMSE").to_csv(out_dir / "rank_by_mean_iMSE.csv", index=False)
    summary_df.sort_values("stability_score").to_csv(out_dir / "rank_by_stability.csv", index=False)
    summary_df.sort_values("overall_score", ascending=False).to_csv(out_dir / "rank_by_overall_score.csv", index=False)

    # Pareto
    summary_df[summary_df["is_pareto"]].sort_values(["mean_iMAE", "mean_iMSE"]).to_csv(
        out_dir / "pareto_only.csv", index=False
    )

    # “过拟合” / 泛化不稳
    overfit_like_df.to_csv(out_dir / "overfit_like_models.csv", index=False)

    # 超参数分析
    importance_df.to_csv(out_dir / "hyperparam_importance.csv", index=False)
    top3_refine_df.to_csv(out_dir / "top3_hyperparams_to_refine.csv", index=False)

    # 每个超参数的 group mean 表
    hp_group_dir = out_dir / "hyperparam_group_tables"
    mkdir(hp_group_dir)
    for hp in HP_NAME_ORDER:
        if hp not in summary_df.columns:
            continue
        if summary_df[hp].dropna().nunique() <= 1:
            continue
        grp = summary_df.groupby(hp, dropna=True).agg(
            count=("exp_name", "count"),
            mean_iMAE=("mean_iMAE", "mean"),
            std_iMAE=("mean_iMAE", "std"),
            mean_iMSE=("mean_iMSE", "mean"),
            std_iMSE=("mean_iMSE", "std"),
            mean_stability=("stability_score", "mean"),
        ).reset_index().sort_values(hp)
        grp.to_csv(hp_group_dir / f"{hp}_group_stats.csv", index=False)

    # 作图
    save_corr_heatmap(summary_df, out_dir)
    save_pareto_plot(summary_df, out_dir)
    save_hyperparam_vs_perf_plots(summary_df, out_dir)
    save_monthly_best_heatmap(monthly_df, out_dir)

    # 文本报告
    save_text_report(out_dir, summary_df, importance_df, overfit_like_df, top3_refine_df)

    # 终端输出
    print("\n[Top 10 by overall_score]", flush=True)
    print(
        summary_df.sort_values("overall_score", ascending=False)[
            ["exp_name", "overall_score", "mean_iMAE", "mean_iMSE", "stability_score"]
        ].head(10).to_string(index=False),
        flush=True
    )

    print("\n[Top 10 by mean_iMAE]", flush=True)
    print(
        summary_df.sort_values("mean_iMAE")[
            ["exp_name", "mean_iMAE", "mean_iMSE", "stability_score"]
        ].head(10).to_string(index=False),
        flush=True
    )

    print("\n[Top 10 by mean_iMSE]", flush=True)
    print(
        summary_df.sort_values("mean_iMSE")[
            ["exp_name", "mean_iMSE", "mean_iMAE", "stability_score"]
        ].head(10).to_string(index=False),
        flush=True
    )

    print("\n[Pareto front]", flush=True)
    print(
        summary_df[summary_df["is_pareto"]].sort_values(["mean_iMAE", "mean_iMSE"])[
            ["exp_name", "mean_iMAE", "mean_iMSE"]
        ].to_string(index=False),
        flush=True
    )

    print("\n[Hyperparameter importance top 10]", flush=True)
    if len(importance_df) > 0:
        print(
            importance_df[
                ["hyperparam", "importance_score", "best_value_by_mean_iMAE", "best_value_by_mean_iMSE"]
            ].head(10).to_string(index=False),
            flush=True
        )

    print("\n[Top 3 hyperparameters to refine next]", flush=True)
    if len(top3_refine_df) > 0:
        print(
            top3_refine_df[
                ["hyperparam", "refine_priority", "best_value_by_mean_iMAE", "best_location", "next_action"]
            ].to_string(index=False),
            flush=True
        )

    print(f"\n[done] outputs saved to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()