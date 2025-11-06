import numpy as np
import math
import matplotlib.pyplot as plt
import time


def triu(A):
    N = A.shape[0]
    U = np.zeros((N, N))
    for i in range(N):
        U[i, i + 1:] = A[i, i + 1:]
    return U


def tril(A):
    N = A.shape[0]
    L = np.zeros((N, N))
    for i in range(N):
        L[i, :i] = A[i, :i]
    return L


def diag(A):
    N = A.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = A[i, i]

    return D


def norm(vec):
    return math.sqrt(np.sum(vec ** 2))


def forward(X, y):
    N = X.shape[0]
    res = np.zeros((N, 1))
    for i in range(N):
        temp = y[i, 0] - np.dot(X[i, :i], res[:i, 0])
        res[i, 0] = temp / X[i, i]
    return res


def backward(X, y):
    N = X.shape[0]
    res = np.zeros((N, 1))
    for i in reversed(range(N)):
        temp = y[i, 0] - np.dot(X[i, i:], res[i:, 0])
        res[i, 0] = temp / X[i, i]
    return res


def lu(A):
    N = A.shape[0]
    U = np.copy(A)
    L = np.zeros((N, N))

    for i in range(N):
        L[i, i] = 1
        for j in range(i + 1, N):
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= U[i, :] * L[j, i]
    return L, U


def Jacobi(A, b):
    N = A.shape[0]
    start = time.time()

    U = triu(A)
    L = tril(A)
    D = diag(A)

    temp = L + U
    M = np.zeros((N, N))
    for i in range(N):
        M[i, :] = -temp[i, :] / D[i, i]

    w = np.zeros((N, 1))
    for i in range(N):
        w[i, 0] = b[i, 0] / D[i, i]

    x = np.zeros((N, 1))
    r = [norm(np.dot(A, x) - b)]
    goal = 1e-9
    limit = 1e9
    while goal < r[-1] < limit:
        x = np.dot(M, x) + w
        r.append(norm(np.dot(A, x) - b))

    end = time.time()

    return r, end - start


def Gauss_Seidel(A, b):
    N = A.shape[0]
    start = time.time()

    U = triu(A)
    T = A - U
    w = forward(T, b)

    x = np.zeros((N, 1))
    r = [norm(np.dot(A, x) - b)]
    goal = 1e-9
    limit = 1e9
    while goal < r[-1] < limit:
        x = -forward(T, np.dot(U, x)) + w
        r.append(norm(np.dot(A, x) - b))

    end = time.time()

    return r, end - start


def LU(A, b):
    start = time.time()
    # s = time.time()
    L, U = lu(A)
    # e = time.time()
    y = forward(L, b)
    x = backward(U, y)

    end = time.time()
    # print("Czas faktoryzacji: " + str(e - s))
    return norm(np.dot(A, x) - b), end - start


def setA1(A, value):
    N = A.shape[0]
    for i in range(N):
        A[i, i] = value


def setA2(A, value):
    N = A.shape[0]
    for i in range(N - 1):
        A[i + 1, i] = value
        A[i, i + 1] = value


def setA3(A, value):
    N = A.shape[0]
    for i in range(N - 2):
        A[i + 2, i] = value
        A[i, i + 2] = value


def ExerciseA(N):
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    for i in range(N):
        b[i, 0] = math.sin((i + 1) * (7 + 1))
    setA1(A, 13)
    setA2(A, -1)
    setA3(A, -1)

    return A, b


def ExerciseB():
    N = 1297
    A, b = ExerciseA(N)
    r1, t1 = Jacobi(A, b)
    r2, t2 = Gauss_Seidel(A, b)

    print("Jacobi: " + str(len(r1)) + " (" + str(t1) + " s) r: " + str(r1[-1]))
    print("Gauss-Seidel: " + str(len(r2)) + " (" + str(t2) + " s) r: " + str(r2[-1]))

    plt.plot(r1, label="Jacobi", color='blue')
    plt.plot(r2, label="Gauss-Seidel", color='green')
    plt.hlines(y=1e-9, xmin=0, xmax=max(len(r1), len(r2)), color='r', linestyle='--', label="Docelowa dokładność")
    plt.yscale('log')
    plt.xlabel('Iteracja')
    plt.ylabel('Norma residuum')
    plt.legend()
    plt.title('Porównanie zbieżności metod: Jacobi vs Gauss-Seidel')
    plt.grid(True)
    plt.show()


def ExerciseC():
    N = 1297
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    for i in range(N):
        b[i, 0] = math.sin((i + 1) * (7 + 1))

    setA1(A, 3)
    setA2(A, -1)
    setA3(A, -1)
    r1, t1 = Jacobi(A, b)
    r2, t2 = Gauss_Seidel(A, b)

    print("Jacobi: " + str(len(r1)) + " (" + str(t1) + " s) r: " + str(r1[-1]))
    print("Gauss-Seidel: " + str(len(r2)) + " (" + str(t2) + " s) r: " + str(r2[-1]))

    plt.plot(r1, label="Jacobi", color='blue')
    plt.plot(r2, label="Gauss-Seidel", color='green')
    plt.hlines(y=1e9, xmin=0, xmax=max(len(r1), len(r2)), color='r', linestyle='--', label="Limit błędu")
    plt.yscale('log')
    plt.xlabel('Iteracja')
    plt.ylabel('Norma residuum')
    plt.legend()
    plt.title('Porównanie zbieżności metod: Jacobi vs Gauss-Seidel')
    plt.grid(True)
    plt.show()


def ExerciseD():
    N = 1297
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    for i in range(N):
        b[i, 0] = math.sin((i + 1) * (7 + 1))
    setA1(A, 3)
    setA2(A, -1)
    setA3(A, -1)
    r, t = LU(A, b)
    print("Residuum: " + str(r))
    print("Czas: " + str(t) + " s")


def ExerciseE():
    N = [100, 500, 1000, 2000, 3000, 4000]
    timeJacobi = []
    timeGauss_Seidel = []
    timeLU = []
    for n in N:
        A, b = ExerciseA(n)
        _, t1 = Jacobi(A, b)
        _, t2 = Gauss_Seidel(A, b)
        _, t3 = LU(A, b)
        print(str(t1) + "   " + str(t2) + "   " + str(t3))
        timeJacobi.append(t1)
        timeGauss_Seidel.append(t2)
        timeLU.append(t3)

    plt.plot(N, timeJacobi, label="Jacobi", color='blue')
    plt.plot(N, timeGauss_Seidel, label="Gauss-Seidel", color='green')
    plt.plot(N, timeLU, label="LU", color='red')
    plt.yscale('log')
    plt.xlabel('Wielkość macierzy')
    plt.ylabel('Czas obliczeń [s]')
    plt.legend()
    plt.title('Porównanie szybkości metod: Jacobiego, Gaussa-Seidela i LU')
    plt.grid(True)
    plt.show()

    plt.plot(N, timeJacobi, label="Jacobi", color='blue')
    plt.plot(N, timeGauss_Seidel, label="Gauss-Seidel", color='green')
    plt.plot(N, timeLU, label="LU", color='red')
    plt.xlabel('Wielkość macierzy')
    plt.ylabel('Czas obliczeń [s]')
    plt.legend()
    plt.title('Porównanie szybkości metod: Jacobiego, Gaussa-Seidela i LU')
    plt.grid(True)
    plt.show()


# ExerciseB()  # Porównanie zbieżności metod Jacobi i Gauss-Seidel dla dobrze uwarunkowanej macierzy (a_ii = 13)
# ExerciseC()  # Porównanie metod Jacobi i Gauss-Seidel dla słabo uwarunkowanej macierzy (a_ii = 3)
# ExerciseD()  # Rozwiązanie układu metodą LU i analiza dokładności oraz czasu wykonania
# ExerciseE()  # Porównanie czasu działania metod Jacobi, Gauss-Seidel i LU dla różnych rozmiarów macierzy

