"""Figura de la formosa: distribución empírica de Lambda_MAF / Lambda* (continua) y de
Lambda(z_B) / Lambda* (discontinua) por régimen.

Lee output/formose_bound.csv (exp_formose_bound.py, tasas con mpmath). Si no existe, lee
output/formose_ensemble.csv (exp_formose.py) y dibuja solo las curvas del MAF.
Escribe output/formose_maf_ratio.pdf. Con --lang es, los rótulos en castellano.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REGIMES = ["NH3", "HCOOH", "agua"]
COLORS = {"NH3": "#2a78d6", "HCOOH": "#eb6834", "agua": "#1baf7a"}
STYLES = {"maf": "-", "B": "--"}
TEXT = {
    "en": {"NH3": "Ammonia", "HCOOH": "Formic acid", "agua": "Water only",
           "maf": r"$\Lambda_{\mathrm{MAF}}/\Lambda^*$", "B": r"$\Lambda(\mathbf{z}_B)/\Lambda^*$",
           "x": r"Rate relative to $\Lambda^*$", "y": "Fraction of realizations"},
    "es": {"NH3": "Amoníaco", "HCOOH": "Ácido fórmico", "agua": "Solo agua",
           "maf": r"$\Lambda_{\mathrm{MAF}}/\Lambda^*$", "B": r"$\Lambda(\mathbf{z}_B)/\Lambda^*$",
           "x": r"Tasa relativa a $\Lambda^*$", "y": "Fracción de realizaciones"},
}


def load_ratios(path):
    """Cocientes por régimen y curva: {curva: {régimen: array ordenado en (0, 1]}}.

    Con formose_bound.csv, curvas "maf" (L_maf / Lstar) y "B" (L_B / Lstar); con
    formose_ensemble.csv, solo "maf" (L_maf / Lstar_LB). Lambda* está en [Lstar, Lstar_UB]
    (cota superior certificada): un cociente mayor que 1 solo puede deberse a esa brecha, y
    se recorta a 1; una tasa mayor que Lstar_UB indicaría un error en los datos.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bound = "L_B" in rows[0]
    lstar = "Lstar" if bound else "Lstar_LB"
    curves = ["maf", "B"] if bound else ["maf"]
    col = {"maf": "L_maf", "B": "L_B"}
    out = {c: {} for c in curves}
    for c in curves:
        for reg in REGIMES:
            R = [r for r in rows if r["regime"] == reg]
            if not R:
                raise ValueError(f"No hay realizaciones del régimen {reg!r} en {path}")
            L = np.array([float(r[col[c]]) for r in R])
            if np.any(L > np.array([float(r["Lstar_UB"]) for r in R])):
                raise ValueError(f"{col[c]} > Lstar_UB en el régimen {reg!r}: revisa {path}")
            out[c][reg] = np.sort(np.minimum(L / np.array([float(r[lstar]) for r in R]), 1.0))
    return out


def ecdf(x):
    """Función de distribución empírica de x, lista para ax.step(xs, ys, where="post").

    x: array ordenado de forma creciente (como los de load_ratios).
    Devuelve (xs, ys): xs repite x[0] al principio con ys = 0, para que la curva arranque
    en 0, y después sube 1/n en cada valor de x. La curva escalonada vale, en cada punto t,
    la fracción de valores de x menores o iguales que t; los valores repetidos dan un único
    salto de altura proporcional a su número.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x no puede estar vacío")
    if np.any(np.diff(x) < 0):
        raise ValueError("x debe estar ordenado de forma creciente")
    n = x.size
    xs = np.concatenate(([x[0]], x))
    ys = np.arange(n + 1) / n
    return xs, ys


class _Stacked(HandlerBase):
    """Muestra de leyenda con varias líneas una encima de otra (una por régimen)."""

    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        k = len(orig)
        return [Line2D([-xdescent, width - xdescent], [height * (k - 0.5 - i) / k - ydescent] * 2,
                       color=h.get_color(), ls=h.get_linestyle(), lw=h.get_linewidth(), transform=trans)
                for i, h in enumerate(orig)]


def plot(ratios, out, lang="en"):
    """Una curva por régimen (color) y cociente (estilo de línea), en escala logarítmica,
    con líneas de referencia en 0.1 y en 1. Dos leyendas: regímenes y cocientes."""
    t = TEXT[lang]
    xmin = min(r.min() for c in ratios.values() for r in c.values())
    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for x_ref in (0.1, 1.0):
            ax.axvline(x_ref, color="0.6", ls=":", lw=0.8, zorder=1)
        for c, by_reg in ratios.items():
            for reg in REGIMES:
                xs, ys = ecdf(by_reg[reg])
                ax.step(xs, ys, where="post", color=COLORS[reg], ls=STYLES[c], lw=1.4, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(10 ** np.floor(np.log10(xmin)), 1.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
        regs = [Line2D([], [], color=COLORS[reg], lw=1.4, label=t[reg]) for reg in REGIMES]
        leg = ax.legend(handles=regs, loc="upper left", frameon=False)
        if len(ratios) > 1:
            ax.add_artist(leg)
            # clave de estilo con las tres curvas de cada cociente, en sus colores
            kinds = [tuple(Line2D([], [], color=COLORS[reg], ls=STYLES[c], lw=1.4) for reg in REGIMES)
                     for c in ratios]
            ax.legend(kinds, [t[c] for c in ratios], handler_map={tuple: _Stacked()}, handleheight=1.6,
                      loc="upper left", bbox_to_anchor=(0.0, 0.70), frameon=False)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def main(lang="en"):
    src = os.path.join(ROOT, "output", "formose_bound.csv")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "output", "formose_ensemble.csv")
        print(f"No existe formose_bound.csv: solo las curvas del MAF ({src})")
    out = os.path.join(ROOT, "output", "formose_maf_ratio.pdf")
    ratios = load_ratios(src)
    for c, by_reg in ratios.items():
        for reg in REGIMES:
            r = by_reg[reg]
            print(f"{c:3s} {TEXT['es'][reg]:14s} n = {r.size}   < 0.1: {(r < 0.1).sum():2d}   "
                  f">= 0.999: {(r >= 0.999).sum():2d}   mínimo: {r.min():.1e}")
    plot(ratios, out, lang)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(TEXT), default="en", help="idioma de los rótulos")
    main(ap.parse_args().lang)
