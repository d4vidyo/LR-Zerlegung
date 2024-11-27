import numpy as np

def LRkompakt(A, **options):
    #Extrahiere epsilon und dtype aus options,
    #Standardwerte falls nicht vorhanden:
    epsilon = options.get('epsilon') or 0
    typ = options.get('dtype') or np.float64

    m = len(A)
    n = len(A[0])
    r = -1
    Stufen = []

    p = np.array([*range(0, m)], dtype = np.int64) 
    #[*range(0, m)] ergibt Liste [0, 1, ... , m-1]
    v = 0
    
    for j in range(0, n):
        spalte = A[r+1:m, j]
        l = np.argmax(abs(spalte)) + r + 1
        if abs(A[l][j]) <= epsilon: continue

        r += 1
        Stufen.append(j)

        if r != l:
            A[[r,l]] = A[[l,r]]
            p[[r,l]] = p[[l,r]]
            v += 1
        for i in range(r+1, m):
            lambda_ir = - A[i][j] / A[r][j]
            A[[i]] += typ(lambda_ir * A[[r]])
            A[i][r] = typ(-lambda_ir)

    return A, p, r, Stufen, v


def LRzerlegung(Aorg, **options):
    typ = options.get('dtype') or np.float64
    A, p, r, Stufen, v = LRkompakt(Aorg.copy(), **options)
    m = len(A)
    n = len(A[0])

    P = np.eye( m , dtype = typ)
    I = np.eye( m , dtype = typ)
    for i in range(0, len(p)):
        P[i] = I[p[i]]

    R = np.zeros( (m, n) , dtype = typ)
    L = np.eye( m , dtype = typ)
    j = 0
    for i in range(n):
        if i in Stufen: j = i
        R[0:j+1 , i] = A[0:j+1 , i]
        if i < m: L[j+1:m , j] = A[j+1:m , j] 

    return R, L, P, r, Stufen, v


def Vorwaertsloesen(L, c, **options):
    typ = options.get('dtype') or np.float64
    m = len(L)
    y = np.zeros(m, dtype = typ)
    for i in range(0,m):
        y[i] = 1/L[i][i] * (c[i] - L[i,:i].dot(y[:i]))
    return y


def Rueckwertsloesen(R, r, Stufen, d, **options):
    epsilon = options.get('epsilon') or 0
    typ = options.get('dtype') or np.float64
    m = len(R)
    n = len(R[0])
    x = np.zeros(n, dtype = typ)
    K = np.zeros((n, n-r), dtype = typ)

    if r < m:
        for i in range(r+1, m):
            if abs(d[i]) > epsilon:
                raise Exception("LGS nicht loesbar!")
                return

    k = 0
    i = r
    for j in range(n-1, -1, -1):
        if j == Stufen[i]:
            x[j] = 1/R[i][j] * (d[i] - R[i,j:n].dot(x[j:n]))

            for q in range(0, n-r):
                Sum = 0
                for l in range(j , n):
                    Sum += R[i][l] * K[l][q]
                K[j][q] = 1/R[i][j] * (-Sum)

            i-=1
        else:
            k += 1
            x[j] = 0
            K[j][k]=1
            
    return x, K
        

def main():
    typ = np.float64
    A = np.array([
        [0,2,4,2],
        [3,3,3,3],
        [1,2,3,1]
        ], dtype=typ)

    b = np.array([4,3,3], dtype=typ)

    epsilon = 0.001 #Testwert

    R, L, P, r, Stufen, v = LRzerlegung(A.copy(), dtype = typ, epsilon = epsilon)
    c = P.dot(b)
    d = Vorwaertsloesen(L, c, dtype = typ, epsilon = epsilon)
    x, K = Rueckwertsloesen(R, r, Stufen, d, dtype = typ, epsilon = epsilon)

    print(x if np.all(abs(A.dot(x) - b) < epsilon) else "Gefundene Loesung erfuellt LGS nicht.")

if __name__ == "__main__":
    main()