"""Run the complete music statistics study without network or a notebook kernel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from generators import conditional_generator, independent_generator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/spotify_tracks_sample.csv"
FIG = ROOT / "figures"

INK = "#18323b"
BLUE = "#2473bd"             # Observed tracks
ORANGE = "#df7546"           # Conditional generator
GREEN = "#31816c"            # Independent control
GRID = "#dce5e5"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelcolor": INK,
        "axes.edgecolor": "#9baeb0", "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.8,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff", "figure.dpi": 140,
    })


def spearman_permutation(x: np.ndarray, y: np.ndarray, repeats=1999, seed=18) -> float:
    """Permutation p-value for H0: no X-Y association (exchangeable pairs)."""
    xr = stats.rankdata(x)
    yr = stats.rankdata(y)
    xr = (xr - xr.mean()) / xr.std(ddof=0)
    yr = (yr - yr.mean()) / yr.std(ddof=0)
    observed = abs(float(np.mean(xr * yr)))
    rng = np.random.default_rng(seed)
    count = sum(abs(float(np.mean(xr * rng.permutation(yr)))) >= observed for _ in range(repeats))
    return (count + 1) / (repeats + 1)


def spearman_interval(x: np.ndarray, y: np.ndarray, repeats=500, seed=19):
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats)
    for i in range(repeats):
        indices = rng.integers(0, len(x), size=len(x))
        vals[i] = stats.spearmanr(x[indices], y[indices]).statistic
    return [round(float(q), 4) for q in np.quantile(vals, [0.025, 0.975])]


def summary_rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {"Величина": label, "N": int(len(frame)), "Среднее": round(float(frame[col].mean()), 4),
         "SD": round(float(frame[col].std(ddof=1)), 4),
         "Медиана": round(float(frame[col].median()), 4),
         "P10": round(float(frame[col].quantile(.10)), 4),
         "P90": round(float(frame[col].quantile(.90)), 4),
         "Минимум": round(float(frame[col].min()), 4),
         "Максимум": round(float(frame[col].max()), 4)}
        for col, label in (("danceability", "X: танцевальность"), ("valence", "Y: позитивность"))
    ]


def visual_eda(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for a, col, label, color in zip(ax,
             ["danceability", "valence"], ["X: танцевальность", "Y: позитивность"], [BLUE, ORANGE]):
        a.hist(df[col], bins=np.linspace(0, 1, 31), color=color, alpha=.8,
               edgecolor="white", linewidth=.6)
        a.axvline(df[col].median(), color=INK, ls="--", lw=1.5,
                  label=f"Медиана = {df[col].median():.3f}")
        a.set(title=label, xlabel="Значение от 0 до 1", ylabel="Число треков", xlim=(0,1))
        a.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(FIG / "01_eda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def visual_dependency(df: pd.DataFrame, rho: float) -> None:
    x = df.danceability.to_numpy()
    y = df.valence.to_numpy()
    groups = pd.qcut(df.danceability, 5, labels=False, duplicates="drop")
    mean = df.groupby(groups, observed=True).valence.agg(["mean", "count", "std"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    hb = ax[0].hexbin(x, y, gridsize=35, extent=(0,1,0,1), cmap="Blues", mincnt=1)
    cb = fig.colorbar(hb, ax=ax[0], fraction=.05, pad=.025)
    cb.set_label("Число треков")
    ax[0].set(title=f"Связь двух свойств: ρ Спирмена = {rho:.3f}",
              xlabel="Танцевальность X", ylabel="Позитивность Y", xlim=(0,1), ylim=(0,1))
    xlabels = [f"Q{i}" for i in range(1, len(mean)+1)]
    yerr = 1.96*mean["std"].to_numpy()/np.sqrt(mean["count"].to_numpy())
    ax[1].bar(xlabels, mean["mean"], color=["#abc9d7", "#84afc7", "#568fae", "#337e9d", BLUE], width=.68)
    ax[1].errorbar(range(len(mean)), mean["mean"], yerr=yerr, fmt="none", color=INK, capsize=3)
    ax[1].set(title="Средняя позитивность по квинтилям X", xlabel="От менее к более танцевальным →",
              ylabel="Средний Y ± 95% CI", ylim=(0,1))
    fig.savefig(FIG / "02_dependence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def visual_generator(train: pd.DataFrame, sim: pd.DataFrame, independent: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for a, col, label in ((ax[0,0], "danceability", "Танцевальность X"),
                          (ax[0,1], "valence", "Позитивность Y")):
        a.hist(train[col], bins=np.linspace(0,1,31), density=True, histtype="step",
               color=BLUE, lw=2, label="Реальные данные")
        a.hist(sim[col], bins=np.linspace(0,1,31), density=True, histtype="step",
               color=ORANGE, lw=2, ls="--", label="Условный генератор")
        a.set(title=label, xlabel="Значение от 0 до 1", ylabel="Плотность", xlim=(0,1))
        a.legend(frameon=False, fontsize=9)
    for sample, name, color in ((train,"Данные",BLUE),(sim,"Условный генератор",ORANGE),
                                (independent,"Независимый контроль",GREEN)):
        bins = np.linspace(0,1,6)
        ids = np.digitize(sample.danceability, bins[1:-1])
        mid = (bins[1:]+bins[:-1])/2
        av = np.array([sample.valence[ids==j].mean() for j in range(5)])
        ax[1,0].plot(mid,av,marker="o",label=name,color=color,lw=2)
    ax[1,0].set(title="Улавливаем ли мы зависимость?",xlabel="Танцевальность X",
                ylabel="Средняя позитивность Y",xlim=(0,1),ylim=(0,1))
    ax[1,0].legend(frameon=False,fontsize=8)
    corrs = [stats.spearmanr(s.danceability,s.valence).statistic
             for s in (train,sim,independent)]
    ax[1,1].bar(["Данные", "Условный", "Независимый"], corrs,
                color=[BLUE,ORANGE,GREEN])
    ax[1,1].axhline(0,color=INK,lw=.8)
    ax[1,1].set(title="Корреляция Спирмена",ylabel="ρ",ylim=(-.15,.7))
    for j,value in enumerate(corrs): ax[1,1].text(j,value+(.02 if value>=0 else -.06),f"{value:.3f}",ha="center")
    fig.savefig(FIG / "03_generator.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    if df.track_id.duplicated().any() or df[["danceability","valence"]].isna().any().any():
        raise ValueError("Data must contain distinct complete tracks")
    x, y = df.danceability.to_numpy(), df.valence.to_numpy()
    rho = float(stats.spearmanr(x,y).statistic)
    p_perm = float(spearman_permutation(x,y))
    confidence = spearman_interval(x,y)
    xbins = pd.qcut(x, q=5, duplicates="drop")
    ybins = pd.qcut(y, q=5, duplicates="drop")
    contingency = pd.crosstab(xbins,ybins)
    chi2,p_chi,dof,expected=stats.chi2_contingency(contingency)
    cramers_v = float(np.sqrt(chi2/(len(df)*min(contingency.shape[0]-1,contingency.shape[1]-1))))
    # Holdout is independent of generator fitting (not an extra external dataset).
    train = df.sample(frac=.8,random_state=37)
    hold = df.drop(train.index)
    generated = conditional_generator(train, len(hold), seed=404, bins=5)
    control = independent_generator(train, len(hold), seed=404)
    gen_metrics = {}
    for label,frame in (("train",train),("holdout",hold),("conditional",generated),("independent",control)):
        gen_metrics[label] = {"N":len(frame),"mean_X":round(float(frame.danceability.mean()),4),
              "mean_Y":round(float(frame.valence.mean()),4),"sd_X":round(float(frame.danceability.std()),4),
              "sd_Y":round(float(frame.valence.std()),4),
              "rho":round(float(stats.spearmanr(frame.danceability,frame.valence).statistic),4)}
    ks_x=stats.ks_2samp(hold.danceability, generated.danceability)
    ks_y=stats.ks_2samp(hold.valence, generated.valence)
    results = {
        "source_records":114000,"clean_unique_records":89740,"sample_n":len(df),
        "random_seed":20261121,"sample_design":"simple random sample without replacement",
        "summary":summary_rows(df),"spearman_rho":round(rho,4),
        "spearman_ci95":confidence,"permutation_repeats":1999,
        "permutation_p":p_perm,"chi2_stat":round(float(chi2),3),
        "chi2_df":int(dof),"chi2_p":float(p_chi),
        "cramers_v":round(cramers_v,4),"min_expected_count":round(float(expected.min()),2),
        "generator_metrics":gen_metrics,
        "ks_holdout_X":{"D":round(float(ks_x.statistic),4),"p":round(float(ks_x.pvalue),4)},
        "ks_holdout_Y":{"D":round(float(ks_y.statistic),4),"p":round(float(ks_y.pvalue),4)},
    }
    visual_eda(df)
    visual_dependency(df,rho)
    visual_generator(train,generated,control)
    (ROOT/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:results[k] for k in ["sample_n","spearman_rho","spearman_ci95","permutation_p","chi2_p","cramers_v"]},ensure_ascii=False,indent=2))
    return results


if __name__ == "__main__":
    run()
