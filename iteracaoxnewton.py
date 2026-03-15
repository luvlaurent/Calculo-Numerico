# Comparação entre:
# 1) Método de Newton
# 2) Método de Iteração Linear (Ponto Fixo)

import numpy as np
import matplotlib.pyplot as plt


# Função cuja raiz queremos encontrar
# f(x) = x^3 - x - 2

def f(x):
    return x**3 - x - 2


# Derivada da função (usada no método de Newton)
def df(x):
    return 3*x**2 - 1


# Função de ponto fixo g(x) obtida de f(x)=0
def g(x):
    return (x + 2)**(1/3)


# Parâmetros do algoritmo

x0 = 1        # chute inicial
tol = 1e-6    # tolerância do erro
max_iter = 20 # número máximo de iterações


# Listas para armazenar resultados

x_newton = []
erro_newton = []

x_iter = []
erro_iter = []


# Método de Newton
# x(k+1) = x(k) - f(x)/f'(x)

x = x0

for k in range(max_iter):

    x_new = x - f(x)/df(x)     # fórmula de Newton
    erro = abs((x_new - x)/x_new)  # erro relativo

    x_newton.append(x_new)
    erro_newton.append(erro)

    if erro < tol:
        break

    x = x_new


# Método de Iteração Linear (Ponto Fixo)
# x(k+1) = g(x(k))

x = x0

for k in range(max_iter):

    x_new = g(x)
    erro = abs((x_new - x)/x_new)

    x_iter.append(x_new)
    erro_iter.append(erro)

    if erro < tol:
        break

    x = x_new


# Tabela de resultados

print("Método de Newton")
print("Iteração | Aproximação | Erro")

for i in range(len(x_newton)):
    print(i+1, x_newton[i], erro_newton[i])


print("\nMétodo de Iteração Linear")
print("Iteração | Aproximação | Erro")

for i in range(len(x_iter)):
    print(i+1, x_iter[i], erro_iter[i])


# Gráfico de convergência

plt.plot(erro_newton, marker='o', label="Newton")
plt.plot(erro_iter, marker='x', label="Iteração Linear")

plt.yscale("log")  # escala log para visualizar erros pequenos

plt.xlabel("Iteração")
plt.ylabel("Erro")
plt.title("Comparação da Convergência dos Métodos")

plt.legend()
plt.grid(True)

plt.show()