"""Constantes cinéticas de la formosa por la teoría del estado de transición (Sección 6)."""
import numpy as np

from auxiliary_functions import readScenario
from network import from_scenario

# Constantes físicas (SI exactas) y temperatura de referencia
KB = 1.380649e-23            # J/K
H = 6.62607015e-34           # J s
R_KCAL = 8.314462618 / 4184  # kcal mol^-1 K^-1 (= 1.987204e-3)
T_REF = 298.15               # K

# Intervalos de barreras (kcal/mol) por régimen catalítico y clase de reacción:
# rango de las barreras de Kua y Tripoli (2024) para cada catalizador de referencia,
# ampliado en +-2 kcal/mol (error sistemático que declaran, p. 5). Rangos sin ampliar:
#   aldólica       (Fig. 5 y Fig. 8):           agua 21.1-24.1, HCOOH 18.3-22.2, NH3 8.2-10.6
#   isomerización  (Fig. 4, Fig. 7 y p. 11):    agua 24.3-29.4, HCOOH 24.1-29.9, NH3 11.7-14.7
#   retroaldólica  (Fig. 10):                   agua 31.5-33.6, HCOOH 27.8-30.0, NH3 21.8-22.2
BARRIERS = {
    "NH3":   {"aldolica": (6.2, 12.6),  "isomerizacion": (9.7, 16.7),  "retroaldolica": (19.8, 24.2)},
    "HCOOH": {"aldolica": (16.3, 24.2), "isomerizacion": (22.1, 31.9), "retroaldolica": (25.8, 32.0)},
    "agua":  {"aldolica": (19.1, 26.1), "isomerizacion": (22.3, 31.4), "retroaldolica": (29.5, 35.6)},
}


def load_formose():
    """Red de la formosa (Müller et al., 2022) sin especies externas, con kappa = 1.

    Devuelve (net, food_rxn): la red con sus nombres de especies y el array ordenado de los
    índices de las reacciones que consumen alguna especie externa (el formaldehído, C1a).
    """
    net, internal = from_scenario("formose")
    S_minus, _ = readScenario("formose")
    external = np.setdiff1d(np.arange(S_minus.shape[0]), internal)
    food_rxn = np.flatnonzero(S_minus[external].sum(axis=0) > 0)
    return net, food_rxn



EXPECTED_COUNTS = {"aldolica": 17, "isomerizacion": 11, "retroaldolica": 10}


def classify_formose(net, food_rxn):
    """Clase de cada reacción de la formosa, según su forma estequiométrica.

      aldolica       C_n + C1 -> C_{n+1}: consume formaldehído (está en food_rxn)
      retroaldolica  C_n -> C_a + C_b:    produce dos moléculas
      isomerizacion  C_n -> C_n:          el resto (un reactivo y un producto)
    Comprueba los recuentos esperados (17, 11 y 10) y lanza ValueError si no coinciden.
    Devuelve un array de m etiquetas.
    """
    food = np.zeros(net.m, dtype=bool)
    food[np.asarray(food_rxn, dtype=int)] = True
    n_products = net.S_plus.sum(axis=0)
    labels = np.where(food, "aldolica",
                      np.where(n_products >= 2, "retroaldolica", "isomerizacion"))
    counts = {c: int((labels == c).sum()) for c in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Clasificación inesperada de la formosa: {counts}")
    return labels

def tst_rate(dG, T=T_REF):
    """Constante de velocidad por la teoría del estado de transición (Eyring):
    k = (k_B T / h) exp(-dG / (R T)).

    dG: barrera de energía libre en kcal/mol (escalar o array); T en K.
    Devuelve k en s^-1 (unimoleculares) o M^-1 s^-1 (bimoleculares, estado estándar 1 M).
    """
    dG = np.asarray(dG, dtype=float)
    return (KB * T / H) * np.exp(-dG / (R_KCAL * T))

def sample_barriers(labels, regime, rng):
    """Barrera de cada reacción, uniforme en el intervalo de su clase para el régimen dado.

    labels: etiquetas de classify_formose; regime: "NH3", "HCOOH" o "agua";
    rng: np.random.Generator. Devuelve un array de m barreras en kcal/mol.
    """
    if regime not in BARRIERS:
        raise ValueError(f"Régimen desconocido: {regime!r}; opciones: {list(BARRIERS)}")
    ranges = BARRIERS[regime]
    unknown = set(labels) - set(ranges)
    if unknown:
        raise ValueError(f"Clases sin intervalo de barreras: {sorted(unknown)}")
    lo = np.array([ranges[c][0] for c in labels])
    hi = np.array([ranges[c][1] for c in labels])
    return rng.uniform(lo, hi)





def formose_kappa(labels, food_rxn, regime, rng, C1=1.0):
    """Constantes efectivas de la formosa para una realización del régimen dado.

    Sortea las barreras (sample_barriers), las convierte con tst_rate y multiplica las
    reacciones con formaldehído (food_rxn) por su concentración C1 (M), que es lo que las
    convierte en reacciones de pseudoprimer orden.
    Devuelve (kappa, dG): kappa en s^-1 y las barreras en kcal/mol.
    """
    if C1 <= 0:
        raise ValueError("La concentración de formaldehído C1 debe ser positiva")
    dG = sample_barriers(labels, regime, rng)
    kappa = tst_rate(dG)
    kappa[np.asarray(food_rxn, dtype=int)] *= C1
    return kappa, dG


def formose_realization(regime, rng, C1=1.0):
    """Red de la formosa con constantes sorteadas para una realización del régimen dado.

    Devuelve (net, dG, labels): la red (FirstOrderNetwork con nombres), las barreras en
    kcal/mol y la clase de cada reacción.
    """
    from network import FirstOrderNetwork
    base, food_rxn = load_formose()
    labels = classify_formose(base, food_rxn)
    kappa, dG = formose_kappa(labels, food_rxn, regime, rng, C1)
    net = FirstOrderNetwork(base.S_minus, base.S_plus, kappa, base.names)
    return net, dG, labels







