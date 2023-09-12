import numpy as np

def LUDecomposition(A:np.array):
    N = A.shape[0]
    A_ = np.copy(A)
    for k in range(N):
        A_[k + 1:, k] /= A_[k, k]
        A_[k + 1:, k + 1:] -= np.outer(A_[k + 1:, k], A_[k, k + 1:])
    np.fill_diagonal(L := np.tril(A_), 1)
    return L, np.triu(A_)

def PLUDecomposition(A:np.array):
    N = A.shape[0]
    A_ = np.copy(A)
    P = np.eye(N, N)
    for k in range(N):
        piv = np.argmax(np.abs(A_[k:, k])) + k
        if piv != k:
            A_[[k, piv]] = A_[[piv, k]]
            P[[k, piv]] = P[[piv, k]]
        if A_[k, k] != 0:
            A_[k + 1:, k] /= A_[k, k]
            A_[k + 1:, k + 1:] -= np.outer(A_[k + 1:, k], A_[k, k + 1:])
    np.fill_diagonal(L := np.tril(A_), 1)
    return P, L, np.triu(A_)

def HouseXY(x:np.array, y:np.array):
    v = x - (np.linalg.norm(x) / np.linalg.norm(y)) * y
    beta = 2  / np.dot(v, v) 
    return v, beta

def GivensXY(x:np.array, y:np.array):
    pass

def House(x:np.array):
    # This implementation handles the non-full rank
    # matrix also
    M = x.shape[0]
    sigma = x[1:M].T @ x[1:M]
    v = x.copy()
    beta = 0
    if sigma == 0:
        if x[0] >= 0:
            beta = 0
        else:
            beta = 2
    else:
        # These calculation is needed for numerical stability
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        # The v[0] ** 2 needed to cancel with the next line of division 
        # in the overall formula
        beta = (2 * v[0] ** 2) / (sigma + v[0] ** 2)
        v /= v[0]
    return v, beta

def HouseholderQR(A:np.array):
    '''
    return : Q, R matrix in QR Decomposition
    '''
    M, N = A.shape
    Q = np.eye(M, M, dtype=float)
    A_ = np.copy(A)
    m = min(N, M - 1)
    for j in range(m):
        v, beta = House(A_[j:M, j])
        A_[j:M, j:N] -= beta * np.outer(v, v) @ A_[j:M, j:N]
        Q[j:, :] -= beta * np.outer(v, v) @ Q[j:, :]

        # H = np.eye(M, M)
        # H[j:M, j:M] -= beta * np.outer(v, v)
        # Q = H @ Q
        # Store the essential part
        # if j < M:
            # A_[j + 1:M, j] = v[1:M - j + 1]
    return Q.T, A_

def HouseBiDia(A:np.array):
    M, N = A.shape
    m = min(N, M - 1)
    A_ = np.copy(A)
    U = np.eye(M, M, dtype=float)
    V = np.eye(N, N, dtype=float)
    for j in range(m):
        u, beta = House(A_[j:M, j])
        A_[j:M, j:N] -= beta * np.outer(u, u) @ A_[j:M, j:N]
        U[j:, :] -= beta * np.outer(u, u) @ U[j:, :]

        # Same result but use more memory and time
        # H = np.eye(M, M, dtype=float)
        # H[j:M, j:M] -= beta * np.outer(u, u)
        # U = H @ U
        # Save the essential part
        # A_[j + 1:M, j] = u[1, M - j + 1]
        if j + 2 < N:
            v, beta = House(A_[j, j + 1:N].T)
            A_[j:M, j + 1:N] -= beta * A_[j:M, j + 1:N] @ np.outer(v, v)
            V[j + 1:, :] -= beta * np.outer(v, v) @ V[j + 1:, :]

            # H = np.eye(N, N, dtype=float)
            # H[j + 1:N, j + 1:N] -= beta * np.outer(v, v)
            # V = H @ V
            # Save the essential part
            # A_[j, j + 2:N] = v[1, N - j]
    return U.T, A_, V

def HouseTriDia(A:np.array):
    N = A.shape[0]
    A_ = np.copy(A)
    Q = np.eye(N, N)
    for k in range(N - 2):
        v, beta = House(A_[k + 1:N, k])
        p = beta * A_[k + 1:N, k + 1:N] @ v
        w = p - ((beta / 2 * p.T @ v) * v)
        A_[k + 1, k] = A_[k, k + 1] = np.sqrt(A_[k + 1:N, k].T @ A_[k + 1:N, k])
        A_[k + 2:N, k] = A_[k, k + 2:N] = np.zeros_like(A_[k + 2:N, k], dtype=float)
        A_[k + 1:N, k + 1:N] -= (np.outer(v, w) + np.outer(w, v))
        Q[k + 1:, :] -= beta * np.outer(v, v) @ Q[k + 1:, :]

        # H = np.eye(N, N)
        # H[k + 1:N, k + 1:N] -= beta * np.outer(v, v)
        # Q = Q @ H
    return Q.T, A_
    
def Givens(a:float, b:float):
    if b == 0:
        c = 1
        s = 0
    else:
        # Prevent overflow or underflow when 
        # calculating r
        if np.abs(b) > np.abs(a):
            tau = - a / b
            s = 1 / np.sqrt(1 + tau ** 2)
            c = s * tau
        else:
            tau = - b / a
            c = 1/ np.sqrt(1 + tau ** 2)
            s = c * tau
        # Since Python handles the big number itself, 
        # we can implement as following
        # r = np.hypot(a, b)
        # c = a / r
        # s = -b / r
    return c, s

def GivensQR(A:np.array) -> np.array:
    '''
    return : Q matrix in QR Decomposition
    '''
    M, N = A.shape
    A_ = np.copy(A)
    Q = np.eye(M, M)
    for j in range(N):
        for i in range(M - 1, j, -1):
            c, s = Givens(A_[i - 1, j], A_[i, j])
            G = np.array([[c, s], [-s, c]])
            A_[i - 1:i + 1, j:] = G.T @ A_[i - 1:i + 1, j:]
            Q[:, i - 1:i + 1] = Q[:, i - 1:i + 1] @ G
    return Q, A_

def GivensBiDia(A:np.array):
    M, N = A.shape
    A_ = np.copy(A)
    U = np.eye(M, M)
    V = np.eye(N, N)
    for j in range(N):
        for i in range(M - 1, j, -1):
            c, s = Givens(A_[i - 1, j], A_[i, j])
            G = np.array([[c, s], [-s, c]])
            A_[i - 1:i + 1, j:] = G.T @ A_[i - 1:i + 1, j:]
            U[:, i - 1:i + 1] = U[:, i - 1:i + 1] @ G
    for j in range(M):
        for i in range(N - 1, j + 1, -1):
            c, s = Givens(A_[j, i - 1], A_[j, i])
            G = np.array([[c, s], [-s, c]])
            A_[j:, i - 1:i + 1] = A_[j:, i - 1:i + 1] @ G
            V[:, i - 1:i + 1] = V[:, i - 1:i + 1] @ G
    return U, A_, V

def GivensTriDia(A:np.array):
    N = A.shape[0]
    A_ = np.copy(A)
    Q = np.eye(N, N)

    return Q, A_

# Some test cases

# A = np.array([
#     [1, 0, 1, 3],
#     [-2, -3, 1, 2],
#     [3, 3, 0, 1],
#     [2, 1, -2, -1],
# ], dtype=float)
# A = np.array([
#     [1, 7, 3],
#     [7, 4, 5],
#     [3, 5, 1]
# ], dtype=float)
# A = np.array([
#     [0.8147, 0.0975, 0.1576],
#     [0.9058, 0.2785, 0.9706],
#     [0.127 , 0.5469, 0.9572],
#     [0.9134, 0.9575, 0.4854],
#     [0.6324, 0.9649, 0.8003],
# ], dtype=float)
# A = np.array([
#     [12, -51, 4],
#     [6, 167, -68],
#     [-4, 24, -41],
#     [-200, 19, 356],
#     [1, 3, 4]
# ], dtype=float)
# A = np.array([
#     [0, 3, 4], 
#     [0, 4, -2], 
#     [2, 1, 2]
# ], dtype = float)
A = np.array([
    [4, 1, 2, -2],
    [1, 2, 0, 1],
    [2, 0, 3, 2],
    [-2, 1, 2, -1],
], dtype=float)
# U, B, V = HouseBiDia(A)
# print(U @ B @ V.T)
U, B, V = GivensBiDia(A)
print(U @ B @ V.T)
Q, T = HouseTriDia(A)
# print(Q)
# print(A)
print(Q @ T @ Q.T)
# Q, R = HouseholderQR(A)
# print(Q)
# print(R)
# print(Q @ R)

# print(Q := HouseholderQR(A=A))
# print(A)
# print(Q @ np.triu(A))
# Q, R = householder_qr(A)
# print(Q @ R)
# Q = GivensQR(A)
# R = np.triu(A)
# print(Q)
# print(R)
# print(Q @ R)
# Q = GivensQR(A)
# print(Q)
# print(A)
# print(Q @ np.triu(A))
