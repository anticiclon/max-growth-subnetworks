import csv
import numpy as np
import matplotlib.pyplot as plt
import os





ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REGIMES = ["NH3", "HCOOH", "agua"]
LABELS = {"NH3": "Ammonia", "HCOOH": "Formic acid", "agua": "Water only"}
COLORS = {"NH3": "#2a78d6", "HCOOH": "#eb6834", "agua": "#1baf7a"}
STYLES = {"NH3": "-", "HCOOH": "-", "agua": "-"}




def load_ratios(path):
    """Lambda_MAF / Lambda* de cada realización de la formosa, por régimen.

    path: CSV de exp_formose.py (columnas regime, Lstar_LB y L_maf).
    Devuelve {régimen: array ordenado de cocientes en (0, 1]}, con los regímenes en el
    orden de REGIMES. Los cocientes se recortan a 1: L_maf no supera Lambda* salvo por
    redondeo (del orden de 1e-16); un exceso mayor indicaría un error en los datos.
    """
    values = {reg: [] for reg in REGIMES}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            values[row["regime"]].append(float(row["L_maf"]) / float(row["Lstar_LB"]))
    ratios = {}
    for reg in REGIMES:
        r = np.array(values[reg])
        if r.size == 0:
            raise ValueError(f"No hay realizaciones del régimen {reg!r} en {path}")
        if np.any(r > 1 + 1e-9):
            raise ValueError(f"Lambda_MAF > Lambda* en el régimen {reg!r}: revisa {path}")
        ratios[reg] = np.sort(np.minimum(r, 1.0))
    return ratios


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



def plot(ratios, out):
    """Distribución empírica de Lambda_MAF / Lambda* por régimen, en escala logarítmica.

    ratios: salida de load_ratios; out: ruta del PDF que se escribe.
    Una curva por régimen (color y estilo de línea distintos, para que se distingan
    también en blanco y negro) y líneas de referencia en 0.1 y en 1.
    """
    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for x_ref in (0.1, 1.0):
            ax.axvline(x_ref, color="0.6", ls=":", lw=0.8, zorder=1)
        for reg in REGIMES:
            xs, ys = ecdf(ratios[reg])
            ax.step(xs, ys, where="post", color=COLORS[reg], ls=STYLES[reg], lw=1.5,
                    label=LABELS[reg], zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$\Lambda_{\mathrm{MAF}}/\Lambda^*$")
        ax.set_ylabel("Fraction of realizations")
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
        ax.legend(loc="upper left", frameon=False)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        
        
        
        
        
        
def main():
    """Figura de la formosa: lee output/formose_ensemble.csv y escribe
    output/formose_maf_ratio.pdf."""
    src = os.path.join(ROOT, "output", "formose_ensemble.csv")
    out = os.path.join(ROOT, "output", "formose_maf_ratio.pdf")
    ratios = load_ratios(src)
    for reg in REGIMES:
        r = ratios[reg]
        print(f"{LABELS[reg]:14s} n = {r.size}   < 0.1: {(r < 0.1).sum():2d}   "
              f">= 0.999: {(r >= 0.999).sum():2d}   mínimo: {r.min():.1e}")
    plot(ratios, out)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()        
        
        
        
        
        
        
        
        
        