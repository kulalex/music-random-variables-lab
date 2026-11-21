"""Interactive bonus assignment: marimo run marimo_app.py.

No Spotify credentials, network access or paid accounts are required.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Музыка и случайность")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy import stats
    from generators import conditional_generator, independent_generator
    return mo, np, pd, plt, Path, stats, conditional_generator, independent_generator


@app.cell
def _(mo):
    mo.md(r"""
    # 🎵 Музыка и случайность
    ### Можно ли по танцевальности песни предугадать её настроение?

    **Две величины:** $X$ — танцевальность (*danceability*), $Y$ — позитивность
    музыкального настроения (*valence*). Обе лежат в интервале $[0,1]$.

    Одна запись — один трек. **Случайный опыт:** выбираем трек из подготовленной
    базы, не зная заранее его характеристик. Значение конкретного трека фиксировано,
    но результат случайного выбора — случайная величина.

    Здесь можно менять жанр, размер синтетической выборки, seed и число интервалов.
    Графики и статистики обновляются автоматически.
    """)
    return


@app.cell
def _(pd, Path):
    dataset = pd.read_csv(Path(__file__).resolve().parent / "data/spotify_tracks_sample.csv")
    return (dataset,)


@app.cell
def _(mo, dataset):
    possible = sorted(dataset.track_genre.value_counts()[lambda s: s >= 50].index.tolist())
    genre = mo.ui.dropdown(["Все жанры"] + possible, value="Все жанры",
                           label="Музыкальный жанр", searchable=True)
    amount = mo.ui.slider(start=200, stop=8000, step=200, value=2400,
                          label="Сколько новых треков сгенерировать")
    seed = mo.ui.slider(start=0, stop=1000, step=1, value=42, label="Seed")
    bins = mo.ui.slider(start=2, stop=8, step=1, value=5,
                        label="Число групп по танцевальности")
    mo.vstack([
        mo.md("## 1. Настройка эксперимента"),
        mo.hstack([genre, amount], justify="start", gap=3),
        mo.hstack([seed, bins], justify="start", gap=3),
    ])
    return genre, amount, seed, bins


@app.cell
def _(dataset, genre, amount, seed, bins, conditional_generator, independent_generator, stats, mo):
    selected = (dataset if genre.value == "Все жанры"
                else dataset.loc[dataset.track_genre == genre.value])
    selected = selected.reset_index(drop=True)
    generated = conditional_generator(selected, amount.value, seed.value, bins.value)
    independent = independent_generator(selected, amount.value, seed.value)
    rho_real = float(stats.spearmanr(selected.danceability, selected.valence).statistic)
    rho_gen = float(stats.spearmanr(generated.danceability, generated.valence).statistic)
    rho_ind = float(stats.spearmanr(independent.danceability, independent.valence).statistic)
    mo.md(f"""
    ## 2. Что получилось?

    | Характеристика | Реальные треки | Условный генератор | Независимый генератор |
    |---|---:|---:|---:|
    | Число треков | {len(selected):,} | {len(generated):,} | {len(independent):,} |
    | Средняя танцевальность | {selected.danceability.mean():.3f} | {generated.danceability.mean():.3f} | {independent.danceability.mean():.3f} |
    | Средняя позитивность | {selected.valence.mean():.3f} | {generated.valence.mean():.3f} | {independent.valence.mean():.3f} |
    | Корреляция Спирмена | **{rho_real:.3f}** | **{rho_gen:.3f}** | **{rho_ind:.3f}** |

    *Главное сравнение — последняя строка. Независимый генератор может сохранить
    отдельные распределения, но разрушает связь между двумя величинами.*
    """)
    return selected, generated, independent, rho_real, rho_gen, rho_ind


@app.cell
def _(mo, plt, selected, generated, independent, np):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    a, b, c = axes
    for _sample, label, _color in ((selected, "Реальные", "#2473bd"),
                                 (generated, "Условный", "#df7546")):
        a.hist(_sample.danceability, bins=np.linspace(0,1,26), density=True,
               histtype="step", lw=2, color=_color, label=label)
        b.hist(_sample.valence, bins=np.linspace(0,1,26), density=True,
               histtype="step", lw=2, color=_color, label=label)
    a.set(xlabel="X от 0 до 1", ylabel="Плотность", title="Танцевальность")
    b.set(xlabel="Y от 0 до 1", title="Позитивность")
    a.legend(frameon=False)
    b.legend(frameon=False)
    names = ["Реальные", "Условный", "Независимый"]
    values = [selected, generated, independent]
    from scipy.stats import spearmanr
    correlations = [spearmanr(v.danceability, v.valence).statistic for v in values]
    c.bar(names, correlations, color=["#2473bd", "#df7546", "#31816c"])
    c.set(ylabel="ρ Спирмена", title="Сохранилась ли зависимость?", ylim=(-.2,1))
    c.tick_params(axis="x", labelrotation=20)
    mo.vstack([mo.md("## 3. Графическое сравнение"), fig])
    return


@app.cell
def _(mo, plt, selected, generated, independent, np):
    fig2, ax2 = plt.subplots(figsize=(10,4), constrained_layout=True)
    edges = np.linspace(0,1,6)
    for _sample, name, _color in ((selected,"Реальные","#2473bd"),
                                (generated,"Условный генератор","#df7546"),
                                (independent,"Независимый контроль","#31816c")):
        group = np.searchsorted(edges[1:-1], _sample.danceability, side="right")
        centers = (edges[:-1]+edges[1:])/2
        means = [_sample.valence[group==j].mean() if np.any(group==j) else np.nan for j in range(5)]
        ax2.plot(centers, means, marker="o", lw=2, color=_color, label=name)
    ax2.set(xlim=(0,1), ylim=(0,1), xlabel="Танцевальность X",
            ylabel="Средняя позитивность Y", title="Как меняется настроение по мере роста танцевальности?")
    ax2.legend(frameon=False)
    mo.vstack([mo.md("## 4. Проверка условной связи"), fig2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Как работает генератор (математически)

    1. Берём $U_1\sim U(0,1)$ и получаем $X=\widehat F_X^{-1}(U_1)$.
    2. Делим реальные наблюдения $X$ на квантильные группы.
    3. Определяем группу для нового $X$, берём независимое $U_2\sim U(0,1)$
       и получаем $Y=\widehat F_{Y\mid B(X)}^{-1}(U_2)$.

    Так мы сохраняем эмпирические распределения и **зависимость на уровне
    квантильных групп**. Это приближённая модель, она не восстанавливает
    конкретные песни или тонкие связи внутри каждой группы.

    В независимом контрольном генераторе оба значения получают из отдельных
    маргинальных распределений без условного шага. Поэтому корреляция в
    среднем стремится к нулю.

    **Ограничение:** исходная база — удобная коллекция треков, а не случайная
    выборка всей музыки мира. Нельзя на основании этих данных утверждать,
    что танцевальный ритм *вызывает* позитивное настроение.
    """)
    return


if __name__ == "__main__":
    app.run()
