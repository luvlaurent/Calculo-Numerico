import numpy as np 
import matplotlib.pyplot as plt  

# 1. Função de Runge
def runge(x):
    return 1.0 / (1.0 + x**2)

# 2. Método de Vandermonde
def interp_vandermonde(x_nodes, y_nodes, x_eval):
    # x_nodes são pontos usados para construir a interpolação
    # x_eval são pontos usados para desenhare calcular a curva interpolada
    # y_nodes são os valores da função nos pontos de interpolação
    
    # Número de pontos de interpolação
    n = len(x_nodes)

    A = np.zeros((n, n))

    # Cada linha i é: [1, x_i, x_i², x_i³, ..., x_i^(n-1)]
    i = 0
    while i < n:
        j = 0
        while j < n:
            A[i, j] = x_nodes[i]**j 
            j += 1
        i += 1

    print("\nSistema de Vandermonde (A * c = y):")
    print(A)

    # y são os valores da função nos pontos dados
    coeffs = np.linalg.solve(A, y_nodes)

    # Avaliação do polinômio nos pontos desejados
    m = len(x_eval) # quantos pontos serão avaliados
    y_eval = np.zeros(m)

    k = 0
    while k < m:
        s = 0.0  # soma acumulada do polinômio
        p = 0 # índice da potência do polinômio
        while p < n:
            # c0 + c1*x + c2*x² + ...
            s += coeffs[p] * (x_eval[k]**p)
            p += 1
        y_eval[k] = s # guarda o valor calculado do polinômio na posição k
        k += 1

    return y_eval

# 3. Método de Newton (diferenças divididas)

def div_diff(x_nodes, y_nodes):
    n = len(x_nodes)

    dd = np.copy(y_nodes).astype(float)

    # Construção da tabela de diferenças divididas
    j = 1
    while j < n:
        i = n - 1
        while i >= j:
            # Fórmula das diferenças divididas
            dd[i] = (dd[i] - dd[i-1]) / (x_nodes[i] - x_nodes[i-j])
            i -= 1
        j += 1

    # dd agora contém os coeficientes do polinômio de Newton
    return dd


def interp_newton(x_nodes, dd, x_eval):
    n = len(x_nodes)
    m = len(x_eval)

    y_eval = np.zeros(m)

    k = 0
    while k < m:
        # Primeiro termo do polinômio
        s = dd[0]

        # Termo acumulado (x - x0)(x - x1)...
        term = 1.0

        i = 1
        while i < n:
            term *= (x_eval[k] - x_nodes[i-1])
            s += dd[i] * term
            i += 1

        y_eval[k] = s
        k += 1

    return y_eval

# 4. Método próprio: Base de Chebyshev 

def interp_chebyshev(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)

    A = np.zeros((n, n))

    i = 0
    while i < n:
        # Vetor com os polinômios de Chebyshev
        T = np.zeros(n)

        # T0(x) = 1
        T[0] = 1.0

        # T1(x) = x
        if n > 1:
            T[1] = x_nodes[i]

        # Recorrência:
        # T_k(x) = 2*x*T_(k-1)(x) - T_(k-2)(x)
        k = 2
        while k < n:
            T[k] = 2 * x_nodes[i] * T[k-1] - T[k-2]
            k += 1

        # Preenche a linha da matriz
        j = 0
        while j < n:
            A[i, j] = T[j]
            j += 1

        i += 1

    print("\nSistema do método próprio (Chebyshev):")
    print(A)

    coeffs = np.linalg.solve(A, y_nodes)

    # Avaliação do polinômio
    m = len(x_eval)
    y_eval = np.zeros(m)

    k = 0
    while k < m:
        T0 = 1.0
        s = coeffs[0]

        if n > 1:
            T1 = x_eval[k]
            s += coeffs[1] * T1

            j = 2
            while j < n:
                T2 = 2 * x_eval[k] * T1 - T0
                s += coeffs[j] * T2

                # Atualiza termos
                T0 = T1
                T1 = T2
                j += 1

        y_eval[k] = s
        k += 1

    return y_eval

# 5. Programa principal

def main():
    # Intervalo de interpolação
    a, b = -1.0, 1.0

    # Número de pontos
    n_nodes = 11

    # Gera nós igualmente espaçados
    x_nodes = np.linspace(a, b, n_nodes)

    # Avalia a função nesses pontos
    y_nodes = runge(x_nodes)

    # Pontos para desenhar o gráfico
    x_fine = np.linspace(a, b, 500)
    y_true = runge(x_fine)

    # Aplicação dos métodos

    y_vand = interp_vandermonde(x_nodes, y_nodes, x_fine)

    dd = div_diff(x_nodes, y_nodes)
    y_newton = interp_newton(x_nodes, dd, x_fine)

    y_cheb = interp_chebyshev(x_nodes, y_nodes, x_fine)

    print("\nDiferença máxima (Vandermonde vs Newton):")
    print(np.max(np.abs(y_vand - y_newton)))

    # GRÁFICOS

    plt.figure(figsize=(12, 8))

    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.plot(x_fine, y_true, 'k-', linewidth=2, label='Função de Runge')
    plt.plot(x_fine, y_vand, 'r--', label='Vandermonde')
    plt.plot(x_fine, y_newton, 'b--', label='Newton')
    plt.plot(x_fine, y_cheb, 'g-', label='Chebyshev')
    plt.plot(x_nodes, y_nodes, 'ko', label='Nós')

    plt.title('Fenômeno de Runge')
    plt.legend()
    plt.grid()

    # Gráfico de erro
    plt.subplot(2, 1, 2)
    plt.semilogy(x_fine, np.abs(y_vand - y_true), 'r--', label='Erro Vandermonde')
    plt.semilogy(x_fine, np.abs(y_newton - y_true), 'b--', label='Erro Newton')
    plt.semilogy(x_fine, np.abs(y_cheb - y_true), 'g-', label='Erro Chebyshev')

    plt.title('Erro (escala log)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("\nCONCLUSÃO")

# Executa o programa
if __name__ == "__main__":
    main()