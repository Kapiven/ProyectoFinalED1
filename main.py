import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm

#Directorios para guardar resultados
RESULTS_DIR = "results"
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

#Estructura para almacenar resultados de los metodos numericos
@dataclass
class MethodResult:
    t: np.ndarray
    y: np.ndarray

# Metodos numericos para resolver sistemas de ecuaciones diferenciales ordinarias (EDOs) de la forma y' = f(t, y).
def euler(f, t0, y0, tf, h):
    """Metodo de Euler explicito para sistemas y' = f(t, y)."""
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y0 = np.asarray(y0, dtype=float)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]), dtype=float)

    return MethodResult(t, y)

# Metodo de Runge-Kutta de cuarto orden para sistemas y' = f(t, y).
def rk4(f, t0, y0, tf, h):
    """Metodo clasico de Runge-Kutta de cuarto orden para sistemas y' = f(t, y)."""
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y0 = np.asarray(y0, dtype=float)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0

    for i in range(n):
        k1 = np.asarray(f(t[i], y[i]), dtype=float)
        k2 = np.asarray(f(t[i] + h / 2, y[i] + h * k1 / 2), dtype=float)
        k3 = np.asarray(f(t[i] + h / 2, y[i] + h * k2 / 2), dtype=float)
        k4 = np.asarray(f(t[i] + h, y[i] + h * k3), dtype=float)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return MethodResult(t, y)

# Funcion para calcular el error L2 entre la solucion exacta y la aproximada en cada instante de tiempo.
def l2_error(exact, approx):
    return np.linalg.norm(exact - approx, axis=1)

# Funcion para guardar los resultados en tablas CSV y calcular errores por componente y error global.
def save_component_table(name, t, exact, euler_y, rk4_y, component_names):
    data = {"t": t}
    for j, label in enumerate(component_names):
        data[f"{label}_exacta"] = exact[:, j]
        data[f"{label}_euler"] = euler_y[:, j]
        data[f"{label}_rk4"] = rk4_y[:, j]
        data[f"error_{label}_euler"] = np.abs(exact[:, j] - euler_y[:, j])
        data[f"error_{label}_rk4"] = np.abs(exact[:, j] - rk4_y[:, j])

    data["error_norma_euler"] = l2_error(exact, euler_y)
    data["error_norma_rk4"] = l2_error(exact, rk4_y)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"), index=False)
    return df

def plot_time_series(name, t, exact, euler_y, rk4_y, labels):
    for j, label in enumerate(labels):
        plt.figure(figsize=(9, 5))
        plt.plot(t, exact[:, j], label="Exacta", linewidth=2)
        plt.plot(t, euler_y[:, j], "--", label="Euler")
        plt.plot(t, rk4_y[:, j], ":", label="RK4", linewidth=2.5)
        plt.xlabel("t")
        plt.ylabel(label)
        plt.title(f"{name}: componente {label}")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{name}_{label}.png"), dpi=180)
        plt.close()


def plot_phase(name, exact, euler_y, rk4_y):
    plt.figure(figsize=(6, 6))
    plt.plot(exact[:, 0], exact[:, 1], label="Exacta", linewidth=2)
    plt.plot(euler_y[:, 0], euler_y[:, 1], "--", label="Euler")
    plt.plot(rk4_y[:, 0], rk4_y[:, 1], ":", label="RK4", linewidth=2.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{name}: plano fase")
    plt.grid(True, alpha=0.35)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{name}_fase.png"), dpi=180)
    plt.close()

# Funcion para generar tablas de convergencia variando el paso h y calculando el error final para cada metodo.
def convergence_table(name, f, exact_fun, t0, y0, tf, hs):
    rows = []
    for h in hs:
        eu = euler(f, t0, y0, tf, h)
        r4 = rk4(f, t0, y0, tf, h)
        exact = exact_fun(eu.t)
        err_eu = np.linalg.norm(exact[-1] - eu.y[-1])
        err_r4 = np.linalg.norm(exact[-1] - r4.y[-1])
        rows.append({"h": h, "error_final_euler": err_eu, "error_final_rk4": err_r4})

    df = pd.DataFrame(rows)
    df["orden_estimado_euler"] = np.nan
    df["orden_estimado_rk4"] = np.nan
    for i in range(1, len(df)):
        ratio_h = df.loc[i - 1, "h"] / df.loc[i, "h"]
        df.loc[i, "orden_estimado_euler"] = np.log(
            df.loc[i - 1, "error_final_euler"] / df.loc[i, "error_final_euler"]
        ) / np.log(ratio_h)
        df.loc[i, "orden_estimado_rk4"] = np.log(
            df.loc[i - 1, "error_final_rk4"] / df.loc[i, "error_final_rk4"]
        ) / np.log(ratio_h)

    df.to_csv(os.path.join(TABLES_DIR, f"{name}_convergencia.csv"), index=False)
    return df

# Funciones para resolver los problemas planteados en el enunciado, generar tablas de resultados y graficos comparativos.
def run_problem_first_order(h=0.1):
    # y' = 0.5 y, y(0) = 1. Solucion: y(t) = exp(0.5t).
    def f(t, y):
        return [0.5 * y[0]]

    def exact_fun(t):
        return np.exp(0.5 * t).reshape(-1, 1)

    eu = euler(f, 0, [1], 5, h)
    r4 = rk4(f, 0, [1], 5, h)
    exact = exact_fun(eu.t)
    df = save_component_table("primer_orden_exponencial", eu.t, exact, eu.y, r4.y, ["y"])
    plot_time_series("primer_orden_exponencial", eu.t, exact, eu.y, r4.y, ["y"])
    conv = convergence_table("primer_orden_exponencial", f, exact_fun, 0, [1], 5, [0.2, 0.1, 0.05, 0.025])
    return df, conv

def run_problem_second_order(h=0.05):

    gamma = 0.2
    omega = 2.0

    # y'' + 0.4y' + 4y = 0
    # y(0)=1, y'(0)=0

    def f(t, y):
        return [
            y[1],
            -2*gamma*y[1] - omega**2*y[0]
        ]

    def exact_fun(t):

        wd = np.sqrt(omega**2 - gamma**2)

        y_exact = np.exp(-gamma*t) * (
            np.cos(wd*t)
            + (gamma/wd)*np.sin(wd*t)
        )

        v_exact = np.exp(-gamma*t) * (
            -gamma*(np.cos(wd*t)+(gamma/wd)*np.sin(wd*t))
            + (-wd*np.sin(wd*t)+gamma*np.cos(wd*t))
        )

        return np.column_stack([y_exact, v_exact])

    eu = euler(f, 0, [1, 0], 10, h)
    r4 = rk4(f, 0, [1, 0], 10, h)

    exact = exact_fun(eu.t)

    df = save_component_table(
        "segundo_orden_oscilador",
        eu.t,
        exact,
        eu.y,
        r4.y,
        ["y", "v"]
    )

    plot_time_series(
        "segundo_orden_oscilador",
        eu.t,
        exact,
        eu.y,
        r4.y,
        ["y", "v"]
    )

    conv = convergence_table(
        "segundo_orden_oscilador",
        f,
        exact_fun,
        0,
        [1,0],
        10,
        [0.2,0.1,0.05,0.025]
    )

    return df, conv


def linear_exact(A, y0):
    y0 = np.asarray(y0, dtype=float)

    def exact_fun(t):
        return np.array([expm(A * ti) @ y0 for ti in t])

    return exact_fun


def run_linear_systems(h=0.02):
    cases = []

    # Sistema 1 del anteproyecto:
    # x' = 3x + 4y, y' = -4x + 3y. Tiene espiral: e^(3t)(cos 4t, -sin 4t).
    A1 = np.array([[3, 4], [-4, 3]], dtype=float)
    y01 = [1, 0]
    cases.append(("sistema_lineal_oscilatorio", A1, y01, 2.0))

    summaries = []
    for name, A, y0, tf in cases:
        def f(t, y, matrix=A):
            return matrix @ y

        exact_fun = linear_exact(A, y0)
        eu = euler(f, 0, y0, tf, h)
        r4 = rk4(f, 0, y0, tf, h)
        exact = exact_fun(eu.t)
        df = save_component_table(name, eu.t, exact, eu.y, r4.y, ["x", "y"])
        plot_time_series(name, eu.t, exact, eu.y, r4.y, ["x", "y"])
        plot_phase(name, exact, eu.y, r4.y)
        conv = convergence_table(name, f, exact_fun, 0, y0, tf, [0.2, 0.1, 0.05, 0.025])
        summaries.append((name, df, conv))

    return summaries


def run_nonlinear_system():
    # Sistema del anteproyecto:
    # x' = lambda*x - (omega + mu*(x^2+y^2))*y
    # y' = lambda*y + (omega + mu*(x^2+y^2))*x
    lam = -0.10
    omega = 1.0
    mu = 0.20
    y0 = [1.0, 1.0]
    tf = 20.0

    def f(t, y):
        x, z = y
        r2 = x * x + z * z
        return [
            lam * x - (omega + mu * r2) * z,
            lam * z + (omega + mu * r2) * x,
        ]

    # Sin solucion analitica elemental para comparar; se usa RK4 con paso muy fino como referencia numerica.
    reference = rk4(f, 0, y0, tf, 0.001)
    hs = [0.1, 0.05, 0.025, 0.0125]
    rows = []

    for h in hs:
        eu = euler(f, 0, y0, tf, h)
        r4 = rk4(f, 0, y0, tf, h)
        ref_indices = (eu.t / 0.001).round().astype(int)
        ref_y = reference.y[ref_indices]
        diff_methods = l2_error(eu.y, r4.y)
        rows.append({
            "h": h,
            "error_final_euler_vs_ref": np.linalg.norm(eu.y[-1] - ref_y[-1]),
            "error_final_rk4_vs_ref": np.linalg.norm(r4.y[-1] - ref_y[-1]),
            "diferencia_final_euler_rk4": diff_methods[-1],
            "radio_final_euler": np.linalg.norm(eu.y[-1]),
            "radio_final_rk4": np.linalg.norm(r4.y[-1]),
        })

        if h == 0.05:
            df = pd.DataFrame({
                "t": eu.t,
                "euler_x": eu.y[:, 0],
                "euler_y": eu.y[:, 1],
                "rk4_x": r4.y[:, 0],
                "rk4_y": r4.y[:, 1],
                "referencia_x": ref_y[:, 0],
                "referencia_y": ref_y[:, 1],
                "diferencia_euler_rk4": diff_methods,
            })
            df.to_csv(os.path.join(TABLES_DIR, "sistema_no_lineal.csv"), index=False)

            plt.figure(figsize=(6, 6))
            plt.plot(ref_y[:, 0], ref_y[:, 1], label="Referencia RK4 h=0.001", linewidth=2)
            plt.plot(eu.y[:, 0], eu.y[:, 1], "--", label="Euler h=0.05")
            plt.plot(r4.y[:, 0], r4.y[:, 1], ":", label="RK4 h=0.05", linewidth=2.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Sistema no lineal: plano fase")
            plt.grid(True, alpha=0.35)
            plt.axis("equal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_DIR, "sistema_no_lineal_fase.png"), dpi=180)
            plt.close()

            plt.figure(figsize=(9, 5))
            plt.plot(eu.t, diff_methods)
            plt.xlabel("t")
            plt.ylabel("||Euler - RK4||")
            plt.title("Sistema no lineal: diferencia entre metodos")
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_DIR, "sistema_no_lineal_diferencia.png"), dpi=180)
            plt.close()

    conv = pd.DataFrame(rows)
    conv["orden_estimado_euler"] = np.nan
    conv["orden_estimado_rk4"] = np.nan
    for i in range(1, len(conv)):
        ratio_h = conv.loc[i - 1, "h"] / conv.loc[i, "h"]
        conv.loc[i, "orden_estimado_euler"] = np.log(
            conv.loc[i - 1, "error_final_euler_vs_ref"] / conv.loc[i, "error_final_euler_vs_ref"]
        ) / np.log(ratio_h)
        conv.loc[i, "orden_estimado_rk4"] = np.log(
            conv.loc[i - 1, "error_final_rk4_vs_ref"] / conv.loc[i, "error_final_rk4_vs_ref"]
        ) / np.log(ratio_h)

    conv.to_csv(os.path.join(TABLES_DIR, "sistema_no_lineal_convergencia.csv"), index=False)
    return conv


def main():
    summaries = []
    summaries.append(("primer_orden_exponencial", *run_problem_first_order()))
    summaries.append(("segundo_orden_oscilador", *run_problem_second_order()))
    for item in run_linear_systems():
        summaries.append(item)
    nonlinear_conv = run_nonlinear_system()

    final_rows = []
    for name, df, conv in summaries:
        final_rows.append({
            "problema": name,
            "error_final_euler": df["error_norma_euler"].iloc[-1],
            "error_final_rk4": df["error_norma_rk4"].iloc[-1],
            "orden_euler_aprox": conv["orden_estimado_euler"].dropna().iloc[-1],
            "orden_rk4_aprox": conv["orden_estimado_rk4"].dropna().iloc[-1],
        })

    final_rows.append({
        "problema": "sistema_no_lineal",
        "error_final_euler": nonlinear_conv["error_final_euler_vs_ref"].iloc[2],
        "error_final_rk4": nonlinear_conv["error_final_rk4_vs_ref"].iloc[2],
        "orden_euler_aprox": nonlinear_conv["orden_estimado_euler"].dropna().iloc[-1],
        "orden_rk4_aprox": nonlinear_conv["orden_estimado_rk4"].dropna().iloc[-1],
    })

    summary = pd.DataFrame(final_rows)
    summary.to_csv(os.path.join(TABLES_DIR, "resumen_errores.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()