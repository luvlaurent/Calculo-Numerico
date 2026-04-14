import numpy as np


def resolver_sistema(A, b, x0, metodo="jacobi", omega=1.0, tol=1e-6, max_iter=100):
    """
    Resolve A*x = b por métodos iterativos.
    metodo: "jacobi", "jacobi_relax", "gs", "sor"
    """

    n = len(b)          # número de incógnitas
    x = x0.copy()       # chute inicial
    k = 0               # contador de iterações

    while k < max_iter:

        x_old = x.copy()
        x_new = np.zeros(n)

        i = 0
        while i < n:

            soma = 0
            j = 0
            while j < n:
                if j != i:
                    soma += A[i][j] * x[j]
                j += 1

            # métodos iterativos
            if metodo == "jacobi":
                x_new[i] = (b[i] - soma) / A[i][i]

            elif metodo == "jacobi_relax":
                novo = (b[i] - soma) / A[i][i]
                x_new[i] = (1 - omega) * x[i] + omega * novo

            elif metodo == "gs":
                x[i] = (b[i] - soma) / A[i][i]

            elif metodo == "sor":
                novo = (b[i] - soma) / A[i][i]
                x[i] = (1 - omega) * x[i] + omega * novo

            i += 1

        # atualização do Jacobi
        if metodo in ["jacobi", "jacobi_relax"]:
            x = x_new.copy()

        # erro entre iterações
        erro = np.linalg.norm(x - x_old)

        print(f"Iteração {k} → erro = {erro:.6f}")

        if erro < tol:
            print("\nConvergiu!")
            break

        k += 1

    return x

# Exemplo: equação de calor

A = np.array([
    [2.0, -1.0, 0.0],
    [-1.0, 2.0, -1.0],
    [0.0, -1.0, 2.0]
])

b = np.array([0.0, 0.0, 0.0])
x0 = np.array([1.0, 1.0, 1.0])

# Análise da matriz

cond = np.linalg.cond(A)
print("\nCondicionamento:", cond)

autovalores_A = np.linalg.eigvals(A)
print("Autovalores de A:", autovalores_A)


# matriz de iteração (Jacobi)
D = np.diag(np.diag(A))
R = A - D
B = np.linalg.inv(D) @ R

rho = max(abs(np.linalg.eigvals(B)))
print("Raio espectral (Jacobi):", rho)

# Execução

print("\n--- Jacobi ---")
sol_jacobi = resolver_sistema(A, b, x0, metodo="jacobi")

print("\n--- Jacobi Relax (ω = 0.8) ---")
sol_jacobi_relax = resolver_sistema(A, b, x0, metodo="jacobi_relax", omega=0.8)

print("\n--- Gauss-Seidel ---")
sol_gs = resolver_sistema(A, b, x0, metodo="gs")

print("\n--- SOR (ω = 1.2) ---")
sol_sor = resolver_sistema(A, b, x0, metodo="sor", omega=1.2)

# Resultados

print("\nResultados:")
print("Jacobi:", sol_jacobi)
print("Jacobi Relax:", sol_jacobi_relax)
print("Gauss-Seidel:", sol_gs)
print("SOR:", sol_sor)
