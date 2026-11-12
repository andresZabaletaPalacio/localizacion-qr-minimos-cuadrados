#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experimento_localizacion.py
===========================
Factorización QR, mínimos cuadrados y localización
Andrés Zabaleta y Sara Zambrano · Álgebra Lineal 2
Departamento de Matemáticas, Facultad de Ciencias
Pontificia Universidad Javeriana, Bogotá

Este programa es el «Experimento en Python» del póster (sección 6). Resuelve el
problema de localización de la sección 5 con la factorización QR de las secciones
3 y 4, implementada desde cero (sin np.linalg.qr), y reproduce la Tabla 1 y la
Figura 5 del póster.

Correspondencia con el póster
-----------------------------
  Póster                                  Aquí
  --------------------------------------  ------------------------------------
  Prop. 5 (linealización): A y b          construir_A, construir_b
  Teorema 3 / Prop. 3: A = QR             qr_gram_schmidt
  Sustitución hacia atrás (sección 4)     sustitucion_hacia_atras
  Teorema 4: R x̂ = Qᵀb                    resolver_con_qr
  Prop. 2 y 4, Teoremas 2 y 4             verificar_teoria
  Tabla 1 y análisis (sección 7)          experimento_monte_carlo,
                                          comprobar_afirmaciones_del_poster
  Figura 5                                figura_localizacion

El experimento
--------------
  · 6 puntos de referencia P1,…,P6 y posición real x_real = (6.5, 3.5).
  · Distancias medidas  d̃_i = ‖x_real − p_i‖ + σ ε_i,  con ε_i ~ N(0, 1).
  · Se arma A (5×2) y b, se factoriza A = QR por Gram–Schmidt y se resuelve
    R x̂ = Qᵀb.  Error de localización:  E = ‖x_real − x̂‖₂.
  · Cinco niveles de ruido σ (más σ = 0: datos exactos), 1000 repeticiones cada uno.

Uso
---
    python experimento_localizacion.py                  # todo, con la semilla del póster
    python experimento_localizacion.py --semilla 7      # otra realización del ruido
    python experimento_localizacion.py --repeticiones 20000
    python experimento_localizacion.py --detalle        # un caso concreto, paso a paso
    python experimento_localizacion.py --sin-figuras    # solo números (sin Matplotlib)
    python experimento_localizacion.py --mostrar        # abre las figuras en pantalla

Requisitos: Python 3.8+, NumPy y Matplotlib (Matplotlib solo para las figuras).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# =============================================================================
# 0. Parámetros del experimento (póster, sección 6)
# =============================================================================
ANCLAS = np.array([[0, 0], [10, 0], [10, 8], [0, 8], [5, 12], [5, -3]],
                  dtype=float)                    # filas = P1, ..., P6
X_REAL = np.array([6.5, 3.5])                     # posición real (desconocida en la vida real)
NIVELES_SIGMA = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)  # σ = 0: datos exactos
REPETICIONES = 1000                               # por cada nivel de ruido

# Semilla del generador aleatorio. Con ella, una ejecución reproduce exactamente
# la Tabla 1 del póster (con otra semilla las cifras cambian algo en la segunda
# o tercera cifra decimal: es la variación normal de un experimento Monte Carlo).
SEMILLA = 3748493

SIGMAS_FIGURA = (0.1, 0.5, 1.0)                   # estimaciones que se dibujan (Figura 5)
SIGMA_CIRCULOS = 0.5                              # σ de las circunferencias dibujadas

# Tabla 1 del póster: σ -> (E medio, E medio/σ). Sirve solo para comparar.
TABLA_POSTER = {0.05: (0.044, 0.89), 0.1: (0.092, 0.92), 0.25: (0.223, 0.89),
                0.5: (0.466, 0.93), 1.0: (0.894, 0.89)}


def fmt_sigma(sigma: float) -> str:
    """σ como se escribe en el póster: 0, 0.05, 0.1, 0.25, 0.5, 1.0."""
    texto = f"{sigma:g}"
    return texto if sigma == 0 or "." in texto else texto + ".0"


# =============================================================================
# 1. Del problema de localización al sistema Ax ≈ b  (póster, sección 5)
# =============================================================================
def distancias_exactas(anclas: np.ndarray, x: np.ndarray) -> np.ndarray:
    """d_i = ‖x − p_i‖ para cada referencia p_i (las filas de `anclas`)."""
    return np.linalg.norm(anclas - x, axis=1)


def distancias_medidas(anclas: np.ndarray, x_real: np.ndarray, sigma: float,
                       epsilon: np.ndarray) -> np.ndarray:
    """Distancias con error:  d̃_i = ‖x_real − p_i‖ + σ ε_i,  con ε_i ~ N(0, 1)."""
    return distancias_exactas(anclas, x_real) + sigma * epsilon


def construir_A(anclas: np.ndarray) -> np.ndarray:
    """Matriz A de la Proposición 5: fila i-ésima = 2 (p_i − p_1)ᵀ, i = 2, …, m.

    Tiene tamaño (m−1)×2. Solo depende de las referencias, no de las mediciones.
    """
    return 2.0 * (anclas[1:] - anclas[0])


def construir_b(anclas: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vector b de la Proposición 5: b_i = d_1² − d_i² + ‖p_i‖² − ‖p_1‖², i = 2, …, m.

    Viene de restar la ecuación de P1 a la de P_i en ‖x‖² − 2 p_iᵀx + ‖p_i‖² = d_i²:
    el término no lineal ‖x‖² se cancela y queda  2 (p_i − p_1)ᵀ x = b_i.
    """
    norma_cuadrada = np.sum(anclas ** 2, axis=1)               # ‖p_i‖² de cada ancla
    return d[0] ** 2 - d[1:] ** 2 + norma_cuadrada[1:] - norma_cuadrada[0]


# =============================================================================
# 2. Factorización QR por Gram–Schmidt  (póster, sección 3)
# =============================================================================
def qr_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Factorización QR reducida A = QR (Teorema 3 y Proposición 3 del póster).

    A es m×n con m ≥ n y columnas linealmente independientes. Devuelve
        Q (m×n) con columnas ortonormales e_1, …, e_n  (QᵀQ = I),
        R (n×n) triangular superior con diagonal positiva.

    Construcción del póster: Gram–Schmidt convierte las columnas v_1, …, v_n de A
    en la lista ortonormal e_1, …, e_n y se toma  R_jk = ⟨v_k, e_j⟩.
    Para k > j esos productos son lo que se resta de v_k; lo que sobra es u_k ⟂ e_j
    y R_kk = ‖u_k‖ > 0. (Con n = 2 columnas, como aquí, esta versión «clásica»
    coincide con la «modificada», que es la preferible cuando n es grande.)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m < n:
        raise ValueError("Se necesita m >= n (más ecuaciones que incógnitas).")
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for k in range(n):
        v_k = A[:, k]
        u_k = v_k.copy()                          # lo que queda de v_k al quitar sus proyecciones
        for j in range(k):
            R[j, k] = Q[:, j] @ v_k               # R_jk = ⟨v_k, e_j⟩
            u_k -= R[j, k] * Q[:, j]              # quita la componente de v_k en la dirección e_j
        R[k, k] = np.linalg.norm(u_k)             # R_kk = ‖u_k‖ = ⟨v_k, e_k⟩ > 0
        if R[k, k] <= tol * np.linalg.norm(v_k):
            raise ValueError(
                f"La columna {k + 1} de A depende de las anteriores (R_{k + 1}{k + 1} = 0): "
                "las referencias están alineadas y la solución no es única.")
        Q[:, k] = u_k / R[k, k]                   # e_k, vector unitario
    return Q, R


def sustitucion_hacia_atras(R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Resuelve R x = c con R triangular superior (póster, sección 4).

        x_n = c_n / R_nn,      x_k = (c_k − Σ_{j>k} R_kj x_j) / R_kk,   k = n−1, …, 1.

    Una incógnita por paso, de x_n a x_1, sin eliminación gaussiana; nunca se divide
    por cero porque la diagonal de R es positiva.
    """
    n = len(c)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (c[k] - R[k, k + 1:] @ x[k + 1:]) / R[k, k]
    return x


# =============================================================================
# 3. Mínimos cuadrados con QR  (póster, sección 4: Teorema 4)
# =============================================================================
def resolver_con_factorizacion(Q: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x̂ a partir de una factorización ya hecha: R x̂ = Qᵀb (Teorema 4)."""
    c = Q.T @ b                                   # c = Qᵀb: coordenadas de P_range(A) b en la base e_k
    return sustitucion_hacia_atras(R, c)


def resolver_con_qr(A: np.ndarray, b: np.ndarray):
    """Resuelve min_x ‖Ax − b‖₂ factorizando A = QR. Devuelve (x̂, Q, R)."""
    Q, R = qr_gram_schmidt(A)
    return resolver_con_factorizacion(Q, R, b), Q, R


def error_localizacion(x_real: np.ndarray, x_est: np.ndarray) -> float:
    """E = ‖x_real − x_est‖₂ (póster, sección 6)."""
    return float(np.linalg.norm(x_real - x_est))


def residuo_ecuaciones_normales(A: np.ndarray, b: np.ndarray, x_est: np.ndarray) -> float:
    """‖Aᵀ(b − A x̂)‖: es 0 exactamente si x̂ cumple las ecuaciones normales AᵀA x̂ = Aᵀb."""
    return float(np.linalg.norm(A.T @ (b - A @ x_est)))


# =============================================================================
# 4. Verificación de la teoría del póster con números concretos
# =============================================================================
def verificar_teoria(anclas: np.ndarray = ANCLAS, x_real: np.ndarray = X_REAL,
                     sigma: float = 0.5, imprimir: bool = True) -> bool:
    """Comprueba numéricamente los resultados del póster para esta A y un b con error.

    Usa su propio generador aleatorio para no alterar el experimento principal.
    Devuelve True si todas las comprobaciones se cumplen.
    """
    rng = np.random.default_rng(12345)
    m_ref = len(anclas)
    A = construir_A(anclas)
    filas, n = A.shape
    d_exactas = distancias_exactas(anclas, x_real)
    b_exacto = construir_b(anclas, d_exactas)
    b = construir_b(anclas, distancias_medidas(anclas, x_real, sigma,
                                               rng.standard_normal(m_ref)))
    Q, R = qr_gram_schmidt(A)
    x_est = resolver_con_factorizacion(Q, R, b)
    P = Q @ Q.T                                    # QQᵀ, proyección sobre range A
    resultados: list[bool] = []

    def anotar(texto: str, cumple: bool, detalle: str = "") -> None:
        resultados.append(bool(cumple))
        if imprimir:
            print(f"  {'✓' if cumple else '✗'} {texto}" + (f"   [{detalle}]" if detalle else ""))

    # --- Sección 5: del modelo al sistema ---------------------------------
    anotar("Ejemplo del póster: la distancia exacta a (0, 0) es √(6.5² + 3.5²) ≈ 7.38",
           abs(d_exactas[0] - 7.38) < 0.005, f"d₁ = {d_exactas[0]:.4f}")
    anotar("A es (m−1)×2 con columnas linealmente independientes",
           A.shape == (m_ref - 1, 2) and np.linalg.matrix_rank(A) == 2,
           f"A es {filas}×{n}, rango {np.linalg.matrix_rank(A)}")
    anotar("Prop. 5: sin error, A x_real = b",
           np.linalg.norm(A @ x_real - b_exacto) < 1e-9,
           f"‖A x_real − b‖ = {np.linalg.norm(A @ x_real - b_exacto):.1e}")
    distancia_a_range = np.linalg.norm(b - P @ b)
    anotar(f"Con error (σ = {sigma}) b ∉ range A: el sistema no tiene solución exacta",
           distancia_a_range > 1e-6, f"‖b − QQᵀb‖ = {distancia_a_range:.3f}")

    # --- Sección 3: factorización QR ---------------------------------------
    anotar("Teorema 3 / Prop. 3: A = QR",
           np.linalg.norm(A - Q @ R) < 1e-10, f"‖A − QR‖ = {np.linalg.norm(A - Q @ R):.1e}")
    anotar("QᵀQ = I  (columnas ortonormales)",
           np.linalg.norm(Q.T @ Q - np.eye(n)) < 1e-12,
           f"‖QᵀQ − I‖ = {np.linalg.norm(Q.T @ Q - np.eye(n)):.1e}")
    anotar("R es triangular superior con diagonal positiva",
           np.allclose(np.tril(R, -1), 0) and np.all(np.diag(R) > 0),
           "diag(R) = " + ", ".join(f"{t:.4f}" for t in np.diag(R)))
    A_cuadrada = A[:n]                             # n×n: las primeras n filas de A (columnas l.i.)
    Q_c, R_c = qr_gram_schmidt(A_cuadrada)
    anotar("Teorema 3 (A cuadrada): Q es unitaria (QᵀQ = QQᵀ = I) y A = QR",
           np.linalg.norm(Q_c.T @ Q_c - np.eye(n)) < 1e-12
           and np.linalg.norm(Q_c @ Q_c.T - np.eye(n)) < 1e-12
           and np.linalg.norm(A_cuadrada - Q_c @ R_c) < 1e-10,
           f"‖QQᵀ − I‖ = {np.linalg.norm(Q_c @ Q_c.T - np.eye(n)):.1e}")
    Q_np, R_np = np.linalg.qr(A)                   # solo para comparar (no se usa en el experimento)
    signos = np.sign(np.diag(R_np))
    anotar("Coincide con np.linalg.qr (salvo signos); por la unicidad del Teorema 3",
           np.linalg.norm(Q - Q_np * signos) < 1e-10 and np.linalg.norm(R - (R_np.T * signos).T) < 1e-10)
    y = rng.standard_normal(n)
    anotar("Prop. 4 (i): ‖Qy‖ = ‖y‖",
           abs(np.linalg.norm(Q @ y) - np.linalg.norm(y)) < 1e-12,
           f"{np.linalg.norm(Q @ y):.6f} = {np.linalg.norm(y):.6f}")
    P_formula = A @ np.linalg.solve(A.T @ A, A.T)  # A (AᵀA)⁻¹ Aᵀ, la proyección sobre range A
    anotar("Prop. 4 (ii): QQᵀ = P_range A",
           np.linalg.norm(P - P_formula) < 1e-10 and np.linalg.norm(P @ P - P) < 1e-12
           and np.linalg.norm(P - P.T) < 1e-12 and np.linalg.norm(P @ A - A) < 1e-10,
           f"‖QQᵀ − A(AᵀA)⁻¹Aᵀ‖ = {np.linalg.norm(P - P_formula):.1e}")
    anotar("…y QQᵀ ≠ I porque m > n",
           abs(np.linalg.norm(P - np.eye(filas)) - np.sqrt(filas - n)) < 1e-10,
           f"‖QQᵀ − I‖ = √(m−n) = {np.linalg.norm(P - np.eye(filas)):.4f}")

    # --- Sección 4: QR y mínimos cuadrados ---------------------------------
    x_normales = np.linalg.solve(A.T @ A, A.T @ b)
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    anotar("Teorema 4: R x̂ = Qᵀb da la misma x̂ que las ecuaciones normales AᵀA x̂ = Aᵀb",
           np.linalg.norm(x_est - x_normales) < 1e-8,
           f"‖x̂_QR − x̂_normales‖ = {np.linalg.norm(x_est - x_normales):.1e}")
    anotar("…y que np.linalg.lstsq",
           np.linalg.norm(x_est - x_lstsq) < 1e-8,
           f"‖x̂_QR − x̂_lstsq‖ = {np.linalg.norm(x_est - x_lstsq):.1e}")
    anotar("Teorema 2: el residuo es ortogonal a las columnas de A (Aᵀ(b − Ax̂) = 0)",
           residuo_ecuaciones_normales(A, b, x_est) < 1e-8,
           f"‖Aᵀ(b − Ax̂)‖ = {residuo_ecuaciones_normales(A, b, x_est):.1e}")
    anotar("Teorema 4: mín ‖Ax − b‖ = ‖b − QQᵀb‖",
           abs(np.linalg.norm(A @ x_est - b) - distancia_a_range) < 1e-9,
           f"{np.linalg.norm(A @ x_est - b):.6f} = {distancia_a_range:.6f}")
    x_prueba = rng.standard_normal(n) * 5          # cualquier x, no solo el óptimo
    izquierda = np.linalg.norm(A @ x_prueba - b) ** 2
    derecha = np.linalg.norm(R @ x_prueba - Q.T @ b) ** 2 + distancia_a_range ** 2
    anotar("Teorema 4 (vía geométrica): ‖Ax − b‖² = ‖Rx − Qᵀb‖² + ‖b − QQᵀb‖²  (Pitágoras)",
           abs(izquierda - derecha) < 1e-9 * max(1.0, izquierda),
           f"{izquierda:.6f} = {derecha:.6f}")

    # --- Proposición 2 y conclusión 3 ---------------------------------------
    alineadas = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
    try:
        qr_gram_schmidt(construir_A(alineadas))
        falla = False
    except ValueError:
        falla = True
    anotar("Prop. 2: con referencias alineadas las columnas de A son dependientes (R_kk = 0)",
           falla, "Gram–Schmidt lo detecta y avisa")
    maxima_diferencia = 0.0
    for _ in range(5):
        b_otro = construir_b(anclas, distancias_medidas(anclas, x_real, sigma,
                                                        rng.standard_normal(m_ref)))
        x_reutilizada = resolver_con_factorizacion(Q, R, b_otro)   # Q y R ya calculadas
        x_refactorizada = resolver_con_qr(A, b_otro)[0]            # factorizando de nuevo
        maxima_diferencia = max(maxima_diferencia, np.linalg.norm(x_reutilizada - x_refactorizada))
    anotar("Conclusión 3: factorizar A una vez y reutilizar Q, R da lo mismo que refactorizar",
           maxima_diferencia < 1e-12, f"diferencia máxima = {maxima_diferencia:.1e}")
    return all(resultados)


# =============================================================================
# 5. Experimento Monte Carlo  (póster, sección 6 y Tabla 1)
# =============================================================================
class NivelRuido(NamedTuple):
    sigma: float
    error_medio: float          # E medio sobre las repeticiones
    error_desv: float           # desviación estándar de E
    max_residuo_normal: float   # máximo de ‖Aᵀ(b − Ax̂)‖ en todas las corridas
    max_norma_b: float          # máximo de ‖b‖ (fija la escala del error de redondeo)

    @property
    def cociente(self) -> float:
        """E medio / σ  (indefinido para σ = 0)."""
        return self.error_medio / self.sigma if self.sigma > 0 else float("nan")


def experimento_monte_carlo(anclas: np.ndarray, x_real: np.ndarray, niveles, repeticiones: int,
                            rng: np.random.Generator) -> list[NivelRuido]:
    """Para cada σ repite `repeticiones` veces: medir con error, resolver y medir E.

    Como A depende solo de las referencias, se factoriza UNA sola vez y se reutiliza
    en todas las repeticiones (conclusión 3 del póster); solo cambia b.
    """
    A = construir_A(anclas)
    Q, R = qr_gram_schmidt(A)                                  # una única factorización
    resultados = []
    for sigma in niveles:
        errores = np.empty(repeticiones)
        residuos = np.empty(repeticiones)
        normas_b = np.empty(repeticiones)
        for k in range(repeticiones):
            # σ = 0 no consume números aleatorios: los datos son exactos
            epsilon = rng.standard_normal(len(anclas)) if sigma > 0 else np.zeros(len(anclas))
            b = construir_b(anclas, distancias_medidas(anclas, x_real, sigma, epsilon))
            x_est = resolver_con_factorizacion(Q, R, b)        # R x̂ = Qᵀb
            errores[k] = error_localizacion(x_real, x_est)
            residuos[k] = residuo_ecuaciones_normales(A, b, x_est)
            normas_b[k] = np.linalg.norm(b)
        desviacion = errores.std(ddof=1) if repeticiones > 1 else 0.0
        resultados.append(NivelRuido(sigma, float(errores.mean()), float(desviacion),
                                     float(residuos.max()), float(normas_b.max())))
    return resultados


def ajuste_error_vs_sigma(resultados: list[NivelRuido]) -> float:
    """Pendiente k de  E ≈ k σ  ajustada por mínimos cuadrados (recta por el origen).

    Es otro problema de mínimos cuadrados, de una sola incógnita: minimiza
    Σ (E_i − k σ_i)²  y da  k = Σ σ_i E_i / Σ σ_i².
    """
    sigma = np.array([r.sigma for r in resultados if r.sigma > 0])
    error = np.array([r.error_medio for r in resultados if r.sigma > 0])
    return float(sigma @ error / (sigma @ sigma))


def imprimir_tabla(resultados: list[NivelRuido], repeticiones: int, comparar_poster: bool) -> None:
    """Tabla 1 del póster (σ, E medio, E medio/σ) más la dispersión y la comparación."""
    print(f"  {'σ':>5} │ {'E medio (' + str(repeticiones) + ')':>15} │ {'E medio/σ':>9} │"
          f" {'desv. est. de E':>15} │ {'máx ‖Aᵀ(b−Ax̂)‖':>16}" + (" │ póster" if comparar_poster else ""))
    print("  " + "─" * 6 + "┼" + "─" * 17 + "┼" + "─" * 11 + "┼" + "─" * 17 + "┼" + "─" * 18
          + ("┼" + "─" * 22 if comparar_poster else ""))
    for r in resultados:
        if r.sigma == 0:
            e_txt, c_txt, d_txt = f"{r.error_medio:.0e}", "—", "—"
        else:
            e_txt, c_txt, d_txt = f"{r.error_medio:.3f}", f"{r.cociente:.2f}", f"{r.error_desv:.3f}"
        linea = (f"  {fmt_sigma(r.sigma):>5} │ {e_txt:>15} │ {c_txt:>9} │ {d_txt:>15} │ "
                 f"{r.max_residuo_normal:>16.1e}")
        if comparar_poster and r.sigma in TABLA_POSTER:
            e_p, c_p = TABLA_POSTER[r.sigma]
            coincide = round(r.error_medio, 3) == e_p and round(r.cociente, 2) == c_p
            linea += f" │ {e_p:.3f} / {c_p:.2f} {'✓' if coincide else '≠'}"
        elif comparar_poster:                                  # fila σ = 0: el póster da 1×10⁻¹⁵
            linea += f" │ 1e-15 / — {'✓' if e_txt == '1e-15' else '≠'}"
        print(linea)


def comprobar_afirmaciones_del_poster(resultados: list[NivelRuido]) -> bool:
    """Contrasta con los números las tres afirmaciones de la sección 7 del póster."""
    sin_ruido = next(r for r in resultados if r.sigma == 0)
    con_ruido = [r for r in resultados if r.sigma > 0]
    k = ajuste_error_vs_sigma(resultados)
    cocientes = [r.cociente for r in con_ruido]
    peor_residuo = max(r.max_residuo_normal for r in resultados)
    # Escala del error de redondeo al evaluar Aᵀ(b − Ax̂): ε‖A‖‖b‖ (ε = 2.2×10⁻¹⁶).
    escala = (np.finfo(float).eps * np.linalg.norm(construir_A(ANCLAS), 2)
              * max(r.max_norma_b for r in resultados))
    pruebas = [
        (f"1. Sin ruido (σ = 0) QR recupera la posición real con error ≈ 10⁻¹⁵",
         sin_ruido.error_medio < 1e-13, f"E = {sin_ruido.error_medio:.1e}"),
        (f"2. El error medio crece casi proporcional a σ: E ≈ {k:.2f} σ (póster: 0.90 σ)",
         abs(k - 0.90) < 0.03,
         f"E medio/σ entre {min(cocientes):.2f} y {max(cocientes):.2f}"),
        (f"3. En cada corrida ‖Aᵀ(b − Ax̂)‖ ≲ {2 * escala:.0e} (≈ 2 ε‖A‖‖b‖, solo redondeo): "
         "el residuo es ortogonal a las columnas de A",
         peor_residuo <= 2 * escala,
         f"máximo observado = {peor_residuo:.1e};  ε‖A‖‖b‖ = {escala:.1e}"),
    ]
    for texto, cumple, detalle in pruebas:
        print(f"  {'✓' if cumple else '✗'} {texto}   [{detalle}]")
    return all(c for _, c, _ in pruebas)


def guardar_csv(resultados: list[NivelRuido], repeticiones: int, ruta: Path) -> None:
    """Guarda la tabla en CSV (se abre directo en Excel)."""
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["sigma", f"E_medio_{repeticiones}", "E_medio_sobre_sigma",
                           "desv_est_E", "max_norma_At_residuo"])
        for r in resultados:
            escritor.writerow([r.sigma, f"{r.error_medio:.6g}",
                               "" if r.sigma == 0 else f"{r.cociente:.4f}",
                               f"{r.error_desv:.6g}", f"{r.max_residuo_normal:.3e}"])


def estimar_realizacion(anclas: np.ndarray, x_real: np.ndarray, epsilon: np.ndarray,
                        sigmas=SIGMAS_FIGURA) -> dict:
    """Una sola realización: los mismos errores ε escalados por cada σ. Devuelve {σ: (x̂, E)}."""
    Q, R = qr_gram_schmidt(construir_A(anclas))
    estimaciones = {}
    for sigma in sigmas:
        b = construir_b(anclas, distancias_medidas(anclas, x_real, sigma, epsilon))
        x_est = resolver_con_factorizacion(Q, R, b)
        estimaciones[sigma] = (x_est, error_localizacion(x_real, x_est))
    return estimaciones


def mostrar_detalle(anclas: np.ndarray, x_real: np.ndarray, sigma: float,
                    epsilon: np.ndarray) -> None:
    """Imprime paso a paso un caso concreto, del sistema Ax ≈ b a la estimación x̂ (n = 2)."""
    d_exactas = distancias_exactas(anclas, x_real)
    d = distancias_medidas(anclas, x_real, sigma, epsilon)
    A = construir_A(anclas)
    b = construir_b(anclas, d)
    Q, R = qr_gram_schmidt(A)
    c = Q.T @ b
    x_est = sustitucion_hacia_atras(R, c)
    r = b - A @ x_est
    print(f"\n    Caso concreto con σ = {fmt_sigma(sigma)} (la realización de la Figura 5)")
    print("    Referencia     distancia exacta   distancia medida")
    for i, ((px, py), d_ex, d_me) in enumerate(zip(anclas, d_exactas, d), start=1):
        print(f"    P{i} = ({px:>2g}, {py:>2g})    {d_ex:>10.4f}         {d_me:>10.4f}")
    print("\n    Sistema Ax ≈ b (Proposición 5):  fila de A = 2(pᵢ − p₁)ᵀ  y  bᵢ = d₁² − dᵢ² + ‖pᵢ‖² − ‖p₁‖²")
    for fila, b_i in zip(A, b):
        print(f"      [{fila[0]:>6.1f} {fila[1]:>6.1f}]   {b_i:>9.3f}")
    print("\n    Factorización A = QR por Gram–Schmidt; R (triangular, diagonal positiva) =")
    for fila in R:
        print(f"      [{fila[0]:>9.4f} {fila[1]:>9.4f}]")
    print(f"    c = Qᵀb = ({c[0]:.4f}, {c[1]:.4f})")
    print("    Sustitución hacia atrás (R x̂ = c):")
    print(f"      x̂₂ = c₂ / R₂₂ = {c[1]:.4f} / {R[1, 1]:.4f} = {x_est[1]:.4f}")
    print(f"      x̂₁ = (c₁ − R₁₂ x̂₂) / R₁₁ = ({c[0]:.4f} − {R[0, 1]:.4f}·{x_est[1]:.4f})"
          f" / {R[0, 0]:.4f} = {x_est[0]:.4f}")
    print(f"    x̂ = ({x_est[0]:.4f}, {x_est[1]:.4f}),  x_real = ({x_real[0]:g}, {x_real[1]:g}),"
          f"  E = {error_localizacion(x_real, x_est):.4f}")
    print(f"    Residuo r = b − Ax̂ = {np.array2string(r, precision=3, suppress_small=True)},"
          f"  ‖r‖ = {np.linalg.norm(r):.3f}")
    print(f"    Aᵀr = ({(A.T @ r)[0]:.1e}, {(A.T @ r)[1]:.1e}) ≈ 0: el residuo es ortogonal "
          "a las columnas de A")


# =============================================================================
# 6. Figuras  (póster: Figura 5 y complemento de la sección 7)
# =============================================================================
def _importar_matplotlib(mostrar: bool):
    """Importa pyplot; sin --mostrar usa un backend sin ventanas (guarda a archivo)."""
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figura_localizacion(anclas: np.ndarray, x_real: np.ndarray, epsilon: np.ndarray,
                        estimaciones: dict, ruta: Path, mostrar: bool = False) -> None:
    """Figura 5 del póster: referencias, circunferencias medidas, x_real y estimaciones.

    `estimaciones` es {σ: (x̂, E)} (ver estimar_realizacion). Los ε son los mismos
    para todos los σ: así se ve cómo, con las mismas «malas mediciones» a mayor
    escala, la estimación se aleja más de la posición real.
    """
    plt = _importar_matplotlib(mostrar)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    azul, azul_osc, naranja, rojo, oro = "#3b6fb6", "#1f4e8c", "#e08a1e", "#b22222", "#f2b01e"
    estilos = {SIGMAS_FIGURA[0]: ("o", azul_osc), SIGMAS_FIGURA[1]: ("s", naranja),
               SIGMAS_FIGURA[2]: ("D", rojo)}

    # Mitad del lado de la ventana de zoom: ±1.5 como en el póster, o más si hace falta.
    zoom = max(1.5, 0.5 * np.ceil(2 * 1.15 * max(e for _, e in estimaciones.values())))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    radios = distancias_medidas(anclas, x_real, SIGMA_CIRCULOS, epsilon)
    for (px, py), radio in zip(anclas, radios):
        ax.add_patch(Circle((px, py), radio, facecolor=azul, alpha=0.10, edgecolor="none"))
        ax.add_patch(Circle((px, py), radio, fill=False, edgecolor=azul, linewidth=1.0))
    ax.plot(anclas[:, 0], anclas[:, 1], "^", color="#0b2a5b", markersize=8, zorder=4)
    for i, (px, py) in enumerate(anclas, start=1):
        desplazamiento = (6, -15) if i == 4 else (6, 5)         # P4 queda bajo la ampliación
        ax.annotate(f"$P_{i}$", (px, py), xytext=desplazamiento, textcoords="offset points",
                    color="#0b2a5b", fontsize=11)

    def dibujar_estimaciones(eje, tam):
        for sigma, (x_est, _) in estimaciones.items():
            marcador, color = estilos[sigma]
            eje.plot([x_real[0], x_est[0]], [x_real[1], x_est[1]], "--", color="gray",
                     linewidth=0.8, zorder=3)
            eje.plot(*x_est, marcador, color=color, markeredgecolor="#333333",
                     markersize=tam, zorder=5)
        eje.plot(*x_real, "*", color=oro, markeredgecolor="#8a5a00", markersize=tam + 5, zorder=6)

    dibujar_estimaciones(ax, 7)
    ax.set_xlim(-3.5, 13.5)
    ax.set_ylim(-4, 13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    zoom_ax = ax.inset_axes([0.015, 0.735, 0.25, 0.25])         # ampliación cerca de x_real
    zoom_ax.set_facecolor("white")
    dibujar_estimaciones(zoom_ax, 6)
    zoom_ax.set_xlim(x_real[0] - zoom, x_real[0] + zoom)
    zoom_ax.set_ylim(x_real[1] - zoom, x_real[1] + zoom)
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    zoom_ax.text(0.04, 0.04, f"zoom ±{zoom:g}", transform=zoom_ax.transAxes, fontsize=7,
                 fontweight="bold", color="#0b2a5b")

    leyenda = [Line2D([], [], marker=estilos[s][0], color=estilos[s][1], linestyle="",
                      markeredgecolor="#333333", label=f"estimación con σ = {fmt_sigma(s)}")
               for s in SIGMAS_FIGURA]
    leyenda.append(Line2D([], [], marker="*", color=oro, linestyle="", markeredgecolor="#8a5a00",
                          markersize=11, label="posición real"))
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4,
              frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    ax.set_title(f"Circunferencias de las distancias medidas (σ = {fmt_sigma(SIGMA_CIRCULOS)})",
                 fontsize=10)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")                # PNG
    fig.savefig(ruta.with_suffix(".svg"), bbox_inches="tight")     # SVG vectorial (para el póster)
    if not mostrar:
        plt.close(fig)


def figura_error_vs_sigma(resultados: list[NivelRuido], repeticiones: int, ruta: Path,
                          mostrar: bool = False) -> None:
    """E medio contra σ, con la recta E ≈ kσ ajustada por mínimos cuadrados."""
    plt = _importar_matplotlib(mostrar)
    con_ruido = [r for r in resultados if r.sigma > 0]
    sigma = np.array([r.sigma for r in con_ruido])
    error = np.array([r.error_medio for r in con_ruido])
    error_estandar = np.array([r.error_desv for r in con_ruido]) / np.sqrt(repeticiones)
    k = ajuste_error_vs_sigma(resultados)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(sigma, error, yerr=error_estandar, fmt="o", color="#1f4e8c", capsize=3,
                label=f"E medio ({repeticiones} repeticiones)")
    xs = np.linspace(0, 1.05 * sigma.max(), 50)
    ax.plot(xs, k * xs, "--", color="#b22222", label=f"E ≈ {k:.2f} σ  (mínimos cuadrados)")
    ax.set_xlabel("σ  (error de las mediciones)")
    ax.set_ylabel("E = ‖x_real − x̂‖")
    ax.set_title("El error crece casi proporcional a σ", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")                # PNG
    fig.savefig(ruta.with_suffix(".svg"), bbox_inches="tight")     # SVG vectorial
    if not mostrar:
        plt.close(fig)


# =============================================================================
# 7. Programa principal
# =============================================================================
def leer_argumentos(argv=None) -> argparse.Namespace:
    carpeta_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    ap = argparse.ArgumentParser(
        description="Experimento de localización con QR y mínimos cuadrados (póster).")
    ap.add_argument("--semilla", type=int, default=SEMILLA,
                    help=f"semilla del generador aleatorio (por defecto {SEMILLA}, la del póster)")
    ap.add_argument("--repeticiones", type=int, default=REPETICIONES,
                    help=f"repeticiones por nivel de σ (por defecto {REPETICIONES})")
    ap.add_argument("--carpeta", default=str(carpeta_base / "resultados"),
                    help="carpeta donde se guardan las figuras y la tabla (CSV)")
    ap.add_argument("--detalle", action="store_true",
                    help="muestra paso a paso un caso concreto (A, b, R, c, x̂ y el residuo)")
    ap.add_argument("--sin-figuras", action="store_true", help="no genera figuras (no usa Matplotlib)")
    ap.add_argument("--mostrar", action="store_true", help="muestra las figuras en pantalla")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = leer_argumentos(argv)
    if hasattr(sys.stdout, "reconfigure"):                     # que σ, ‖ y ✓ se vean bien en Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print(" Factorización QR, mínimos cuadrados y localización  ·  experimento del póster")
    print("=" * 78)
    print("Referencias P1…P6:", ", ".join(f"({x:g}, {y:g})" for x, y in ANCLAS))
    print(f"Posición real x_real = ({X_REAL[0]:g}, {X_REAL[1]:g})   "
          f"semilla = {args.semilla}   repeticiones = {args.repeticiones}")

    print("\n[1] Verificación de la teoría del póster con números (un b con error σ = 0.5)")
    teoria_ok = verificar_teoria()
    print("    →", "todas las comprobaciones se cumplen" if teoria_ok else "¡HAY COMPROBACIONES QUE FALLAN!")

    print(f"\n[2] Experimento: {args.repeticiones} repeticiones por nivel de σ  (Tabla 1)")
    rng = np.random.default_rng(args.semilla)
    resultados = experimento_monte_carlo(ANCLAS, X_REAL, NIVELES_SIGMA, args.repeticiones, rng)
    es_la_del_poster = args.semilla == SEMILLA and args.repeticiones == REPETICIONES
    imprimir_tabla(resultados, args.repeticiones, comparar_poster=es_la_del_poster)
    if not es_la_del_poster:
        print("  (semilla o repeticiones distintas a las del póster: las cifras pueden variar "
              "un poco)")

    print("\n[3] Sección 7 del póster: ¿lo que se afirma se ve en los números?")
    comprobar_afirmaciones_del_poster(resultados)

    carpeta = Path(args.carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    guardar_csv(resultados, args.repeticiones, carpeta / "tabla_1_error_localizacion.csv")
    archivos = ["tabla_1_error_localizacion.csv"]

    # Una sola realización de los errores ε (la de la Figura 5): los mismos ε para todos los σ.
    epsilon = rng.standard_normal(len(ANCLAS))
    estimaciones = estimar_realizacion(ANCLAS, X_REAL, epsilon)
    print("\n[4] Una sola realización (Figura 5): los mismos errores ε, escalados por σ")
    for sigma, (x_est, e) in estimaciones.items():
        print(f"    σ = {fmt_sigma(sigma):<4}  x̂ = ({x_est[0]:.4f}, {x_est[1]:.4f})   E = {e:.4f}")
    if args.detalle:
        mostrar_detalle(ANCLAS, X_REAL, SIGMA_CIRCULOS, epsilon)

    if not args.sin_figuras:
        try:
            figura_localizacion(ANCLAS, X_REAL, epsilon, estimaciones,
                                carpeta / "figura_5_localizacion.png", args.mostrar)
            figura_error_vs_sigma(resultados, args.repeticiones,
                                  carpeta / "error_vs_sigma.png", args.mostrar)
            archivos += ["figura_5_localizacion.png (y .svg)", "error_vs_sigma.png (y .svg)"]
            if args.mostrar:
                import matplotlib.pyplot as plt
                plt.show()
        except ImportError:
            print("\nMatplotlib no está instalado: se omiten las figuras "
                  "(instálalo con: pip install matplotlib).")

    print(f"\nArchivos guardados en: {carpeta}")
    for nombre in archivos:
        print("   ", nombre)
    return 0 if teoria_ok else 1


if __name__ == "__main__":
    sys.exit(main())
