import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

# Fungsi untuk mengonversi persamaan string ke fungsi numerik
def parse_equation(equation_str):
    x = sp.symbols('x')
    equation = sp.sympify(equation_str)
    f = sp.lambdify(x, equation, 'numpy')  # Mengubah persamaan menjadi fungsi numpy
    f_prime = sp.lambdify(x, sp.diff(equation, x), 'numpy')  # Turunan dari persamaan
    return f, f_prime

# Metode Newton-Raphson
def newton_raphson(equation_str, x0, tolerance=0.01, max_iter=100):
    f, f_prime = parse_equation(equation_str)
    x = x0  # Titik awal
    iteration = 0
    result = []
    while iteration < max_iter:
        fx = f(x)
        fpx = f_prime(x)

        # Cek pembagi (derivatif) agar tidak menjadi 0
        if fpx == 0:
            result.append("Turunan fungsi f'(x) sama dengan 0. Metode gagal.")
            return result

        # Newton-Raphson formula
        x_new = x - fx / fpx

        # Menampilkan hasil iterasi
        result.append(f"Iterasi {iteration+1}: x = {x_new:.6f}, f(x) = {fx:.6f}")

        # Cek konvergensi
        if abs(x_new - x) < tolerance:
            result.append(f"Konvergen setelah {iteration+1} iterasi.")
            result.append(f"Akar persamaan adalah: {x_new:.6f}")
            return result

        # Update x untuk iterasi berikutnya
        x = x_new
        iteration += 1

    result.append(f"Metode tidak konvergen setelah {max_iter} iterasi.")
    return result

# Fungsi untuk menghitung basis polinom Lagrange
def lagrange_basis(x_values, i, x):
    n = len(x_values)
    basis = 1
    for j in range(n):
        if j != i:
            basis *= (x - x_values[j]) / (x_values[i] - x_values[j])
    return basis

# Fungsi untuk interpolasi Lagrange
def interpolasi_lagrange(x_values, y_values, x):
    n = len(x_values)
    y = 0
    langkah_perhitungan = []

    for i in range(n):
        basis = lagrange_basis(x_values, i, x)
        langkah_perhitungan.append(f"L_{i}({x}) = {basis:.4f}")
        y += y_values[i] * basis

    return y, langkah_perhitungan

# Fungsi untuk plot grafik interpolasi Lagrange
def plot_lagrange(x_values, y_values, x_pred, y_pred):
    x_min, x_max = min(x_values), max(x_values)
    x_plot = np.linspace(x_min, x_max, 500)
    y_plot = []

    for x in x_plot:
        y, _ = interpolasi_lagrange(x_values, y_values, x)
        y_plot.append(y)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label="Interpolasi Lagrange", color="blue", linewidth=2)
    plt.scatter(x_values, y_values, color="red", label="Titik Data", s=100, zorder=5)
    plt.scatter(x_pred, y_pred, color="green", label=f"Prediksi (x={x_pred}, y={y_pred:.4f})", s=150, edgecolors='black', zorder=6)
    plt.title("Grafik Interpolasi Lagrange", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Menu Opsi Antara Metode Interpolasi Lagrange dan Newton-Raphson
def main():
    st.title("Aplikasi Interpolasi dan Newton-Raphson")

    # Pilihan metode
    method = st.sidebar.selectbox(
        "**Pilih Metode:**",
        ("Interpolasi Lagrange", "Newton-Raphson")
    )

    # Jika memilih metode interpolasi Lagrange
    if method == "Interpolasi Lagrange":
        st.write("### Interpolasi Lagrange")

        n = st.sidebar.number_input("Masukkan jumlah titik data:", min_value=2, step=1, value=3)
        x_values = []
        y_values = []
        for i in range(n):
            x = st.sidebar.number_input(f"Masukkan nilai x{i}:", key=f"x{i}", format="%.2f")
            y = st.sidebar.number_input(f"Masukkan nilai y{i}:", key=f"y{i}", format="%.2f")
            x_values.append(x)
            y_values.append(y)

        x = st.sidebar.number_input("Masukkan nilai x yang ingin diperkirakan:", key="x_pred", format="%.2f")

        if x_values and y_values and x:
            # Menampilkan tabel input
            data = {'x': x_values, 'y': y_values}
            df = pd.DataFrame(data)
            st.subheader("Tabel Hasil Input Data")
            st.write(df)

            # Lakukan interpolasi
            y, langkah_perhitungan = interpolasi_lagrange(x_values, y_values, x)

            st.subheader("Hasil Interpolasi")
            st.write(f"Nilai y pada x = {x} adalah **{y:.4f}**")

            st.subheader("Langkah Perhitungan")
            for langkah in langkah_perhitungan:
                st.write(langkah)

            st.subheader("Grafik Interpolasi")
            plot_lagrange(x_values, y_values, x, y)

    # Jika memilih metode Newton-Raphson
    elif method == "Newton-Raphson":
        st.write("### Metode Newton-Raphson")

        equation_str = st.sidebar.text_input('Masukkan persamaan f(x):', '20*x**4 - 2*x**3 + 2*x - 35')
        x0 = st.sidebar.number_input('Titik Awal x0:', value=1.0)
        tolerance = st.sidebar.number_input('Toleransi:', value=0.01)
        max_iter = st.sidebar.number_input('Maksimal Iterasi:', value=100, step=1)

        # Perhitungan Newton-Raphson langsung setelah input
        result = newton_raphson(equation_str, x0, tolerance, max_iter)

        # Menampilkan hasil iterasi
        for line in result:
            st.write(line)

        if len(result) > 2 and 'Akar persamaan' in result[-1]:
            root = float(result[-1].split()[-1])
            f, _ = parse_equation(equation_str)
            x_vals = np.linspace(x0 - 2, x0 + 2, 400)
            y_vals = f(x_vals)

            plt.plot(x_vals, y_vals, label='f(x)')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.scatter(root, f(root), color='red', label=f'Root at x = {root:.4f}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Metode Newton-Raphson')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

if __name__ == "__main__":
    main()
