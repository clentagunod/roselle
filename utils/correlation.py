

import pandas as pd
from scipy import stats

INPUT_CSV = "data/final_dataset/final_docking_survivors.csv"
OUTPUT_CSV = "data/rq4_correlation_result.csv"
COL_A = "ml_composite_score"
COL_B = "mean_docking_affinity"
ALPHA = 0.05


def check_normality(series: pd.Series, label: str) -> tuple:
    stat, p = stats.shapiro(series)
    is_normal = p > ALPHA
    print(f"Shapiro-Wilk ({label}): W={stat:.4f}, p={p:.4f} -> "
          f"{'normal' if is_normal else 'not normal'}")
    return is_normal, p


def main():
    df = pd.read_csv(INPUT_CSV)
    paired = df[[COL_A, COL_B]].dropna()
    a, b = paired[COL_A], paired[COL_B]
    n = len(paired)
    print(f"N = {n}\n")

    if n < 3:
        raise ValueError("Need at least 3 paired observations to test correlation.")
    if n < 20:
        print("NOTE: small sample size -- normality test and correlation both "
              "have low power here. Interpret cautiously.\n")

    normal_a, p_a = check_normality(a, COL_A)
    normal_b, p_b = check_normality(b, COL_B)
    both_normal = normal_a and normal_b

    if both_normal:
        coef, p_value = stats.pearsonr(a, b)
        method = "Pearson r"
    else:
        coef, p_value = stats.spearmanr(a, b)
        method = "Spearman rho"

    tau, tau_p = stats.kendalltau(a, b)

    print(f"\nMethod selected: {method} (both variables normal: {both_normal})")
    print(f"{method} = {coef:.4f}, p-value = {p_value:.4f}")
    significant = p_value < ALPHA
    print(f"Statistically significant at alpha={ALPHA}: {significant}")
    print(f"\nSecondary check -- Kendall's tau = {tau:.4f}, p-value = {tau_p:.4f}")

    result = pd.DataFrame([{
        "n": n,
        "shapiro_p_ml_composite_score": round(p_a, 4),
        "shapiro_p_mean_docking_affinity": round(p_b, 4),
        "both_normal": both_normal,
        "method": method,
        "coefficient": round(coef, 4),
        "p_value": round(p_value, 4),
        "significant_at_0.05": significant,
        "kendall_tau": round(tau, 4),
        "kendall_p_value": round(tau_p, 4),
    }])
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()