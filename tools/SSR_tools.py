import numpy as np
from math import sqrt

NQ = 3

def get_D(d, n):
    '''
    when using simplex method to generate points, we add also reflected ones,
    so the overall number of points created is 2 * (n+1) + 1, where 1 comes
    from the weight of function at zero. Since we add f(0) directly to the
    integral, we don't need to account for it in D here.
    '''
    D = int(n * (2 * (d+1)))
    return D

def radius(d, n):
    rm = d + 2
    D = int(get_D(d, n) / 2)
    r = np.sqrt(2 * np.random.gamma(rm/2., 1., D))
    return r

def batch_simplex_matvec(x):
    nobj = x.shape[1]
    d = x.shape[0]
    mp = d + 1
    r = np.empty((d+1, nobj))
    rv = np.empty(d)
    s = np.zeros(nobj)
    for i in range(d):
        rv[i] = sqrt(mp / ((d-i) * d * (d-i+1.)))
        for o in range(nobj):
            rvo = rv[i] * x[i, o]
            r[i, o] = s[o] + rvo * (d-i)
            s[o] += -rvo
    for o in range(nobj):
        r[d, o] = s[o]
    return r

def butterfly_params(n, k):
    h = int(np.ceil(k))
    log = np.log2(n)
    next_power = 2**int(np.ceil(log))
    cos = np.empty((h, next_power-1))
    sin = np.empty((h, next_power-1))
    perm = np.empty((h, n), np.int32)
    for i in range(h):
        c, s = cos_sin(n)
        cos[i] = c
        sin[i] = s
        p = np.arange(n)
        np.random.shuffle(p)
        perm[i] = p
    return cos, sin, perm

def butterfly_angles(u):
    '''
    Computes angles (in radians) from components of the generating vector
        `u` for random butterfly orthogonal matrices.

    Args:
    =====
    u: a generating vector for random butterfly orthogonal matrices, use
        make_generating_vector() to obtain one.

    Returns:
    ========
    thetas: an 1-D array of angles for computing random butterfly orthogonal
        matrices.
    '''
    thetas = np.arctan2(u[:-1], u[1:])
    return thetas

def cos_sin(N):
    c = np.log2(N)
    f = np.ceil(c)
    n = int(2 ** f)
    u = butterfly_generating_vector(n)
    thetas = butterfly_angles(u)
    if c != f:
        thetas = np.concatenate((thetas[:N-1], np.array([0.] * (n - N))))
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    return cos, sin

def butterfly_generating_vector(n):
    '''
    Generates a vector `u` used to construct random butterfly orthogonal
        matrices.

    Args:
    =====
    n: size of generating vector, n = N + 1 to construct random
        butterfly orthogonal matrix of size N x N.

    Returns:
    ========
    u: generating vector used to calculate angles for random butterfly
        orthogonal matrices.
    '''
    l = n // 2 - 1
    r = np.random.rand(n-1)
    u = np.zeros(n)

    for i in range(l):
        m = n - 2*i
        s = np.sin(2 * np.pi * r[m-2])
        c = np.cos(2 * np.pi * r[m-2])
        pos = n - 2*np.arange(1, i+1) - 1
        ds = 1. / (pos + 1)
        p = np.prod(r[pos]**ds)

        u[m - 1] = np.sqrt(1. - r[m-3]**(2./(m-3))) * p * s
        u[m - 2] = np.sqrt(1. - r[m-3]**(2./(m-3))) * p * c

    s = np.sin(2 * np.pi * r[0])
    c = np.cos(2 * np.pi * r[0])
    pos = n - 2*np.arange(1, l+1) - 1
    ds = 1. / (pos + 1)

    p = np.prod(r[pos]**ds)
    if n % 2 == 0:
        u[0] = c * p
        u[1] = s * p
    else:
        u[2] = (2 * r[1] - 1) * p
        u[1] = 2 * np.sqrt(r[1] * (1 - r[1])) * p * s
        u[0] = 2 * np.sqrt(r[1] * (1 - r[1])) * p * c
    return u





def batch_factor_matvec(x, n, cos, sin):
    '''
    Matvec for Q_n^T x, where n is the index number of n'th factor
        of butterfly matrix Q. Facilitates fast butterfly simplex weights
        multiplication by data vector:
        [QV]^T x = V^T Q^T x = V^T Q_{log2(d)}^T ... Q_0^T x

    Args:
    =====
    x: a batch of data vectors
    n: the index number of n'th butterfly factor Q_n
    cos: cosines used to generate butterfly matrix Q
    sin: sines used to generate butterfly matrix Q
    '''
    nobj = x.shape[1]
    N = x.shape[0]
    d = len(cos) + 1
    blockn = 2 ** n
    nblocks = int(np.ceil(N/blockn))
    r = np.copy(x)
    step = blockn // 2
    for i in range(nblocks - 1):
        shift = blockn*i
        idx = step + shift - 1
        c = cos[idx]
        s = sin[idx]
        for j in range(step):
            i1 = shift + j
            i2 = i1 + step
            for o in range(nobj):
                y1 = x[i1, o]
                y2 = x[i2, o]
                r[i1, o] = c * y1 + s * y2
                r[i2, o] = -s * y1 + c * y2

    # Last block is special since N might not be a power of 2,
    # which causes cutting the matrix and replacing some elements
    # with ones.
    # We calculate t - the number of lines to fill in before proceeding.
    i = nblocks - 1
    shift = blockn * i
    idx = step + shift - 1
    c = cos[idx]
    s = sin[idx]
    t = N - shift - step
    for j in range(t):
        i1 = shift + j
        i2 = i1 + step
        for o in range(nobj):
            y1 = x[i1, o]
            y2 = x[i2, o]
            r[i1, o] = c * y1 + s * y2
            r[i2, o] = -s * y1 + c * y2
    return r


def batch_butterfly_matvec(x, cos, sin, p):
    '''
    Apply butterfly matvec NQ times.
    '''
    d = x.shape[0]
    h = int(np.ceil(np.log2(d)))

    for _ in range(NQ):
        for n in range(1, h+1):
            x = batch_factor_matvec(x, n, cos, sin)
        x = x[p, :]
    return x


def get_batch_mx(x, cos, sin, perm):
    b = batch_butterfly_matvec(x, cos, sin, perm)
    Mx = batch_simplex_matvec(b)
    return Mx

def butt_quad_mapping(x, n, r=None, b_params=None, even=False):
    '''
              |V^T Q_0^T x|
    Mx = \rho |    ...    |
              |V^T Q_n^T x|
    Args:
    =====
    x: the data vector of dimension d.
    n: the parameter defining the new number of features.

    Returns:
    ========
    Mx: the mapping Mx.
    w: the weights.
    '''
    nobj = x.shape[1]
    d = x.shape[0]
    D = get_D(d, n)
    Mx = np.empty((D, nobj))
    if even:
        t = int(np.ceil(2*n))
    else:
        t = int(np.ceil(n))
    if r is None:
        r = radius(d, t)
    if b_params is None:
        b_params = butterfly_params(d, t)

    w = np.empty(D)
    cos, sin, perm = b_params
    if n < 1:
        if n > 0.5:
# TODO check indexing! d+1?
            d0 = int(get_D(d, 1)/2)
            rr = np.ones((d0, 1))
            rr[:, 0] = r[:d0]
            Mx[:d+1, :] = rr * get_batch_mx(x, cos[0], sin[0], perm[0])
            w[:] = sqrt(d) / rr[:, 0]
            if even:
                dd = d0
                d0 = r.shape[0]
                rr = np.ones((d0-dd, 1))
                rr[:, 0] = r[d0:]
                Mx[d+1:, :] = rr * get_batch_mx(x, cos[1], sin[1],
                                                perm[1])[D-(d+1), :]
                w[d+1:] = sqrt(d) / rr[:, 0]
            else:
                Mx[d+1:, :] = -Mx[:D-(d+1), :]
        else:
            Mx[:, :] = r[:] * get_batch_mx(x, cos[0], sin[0], perm[0])[:D, :]
            w[:] = sqrt(d) / r[:]
        return Mx, w

    if even:
        dd = 0
        for i in range(t-1):
            d0 = int(get_D(d, i+1) / 2)
            rr = np.ones((d0-dd, 1))
            rr[:, 0] = r[dd:d0]
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / rr[:, 0]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    rr * get_batch_mx(x, cos[i], sin[i], perm[i])
            dd = d0

        i = t - 1
        d0 = int(get_D(d, i+1) / 2)
        rr = np.ones((d0-dd, 1))
        rr[:, 0] = r[dd:d0]
        Mx[i*(d+1):, :] = rr * get_batch_mx(x, cos[i], sin[i], perm[i])
        w[i*(d+1):] = sqrt(d) / rr[:, 0]

    else:
        dd = 0
        for i in range(t):
            d0 = int(get_D(d,(i+1)) / 2)
            rr = np.ones((d0-dd, 1))
            rr[:, 0] = r[dd:d0]
            w[i*(d+1):(i+1)*(d+1)] = sqrt(d) / rr[:, 0]
            Mx[i*(d+1):(i+1)*(d+1), :] = \
                    rr * get_batch_mx(x, cos[i], sin[i], perm[i])
            dd = d0
        div = t * (d+1)
        Mx[div:, :] = -Mx[:D-div, :]
        w[div:] = w[:D-div]
    return Mx, w


def kernel_SSR(X, Y, n, sigma, **kwargs):
    d = X.shape[1]
    Xc = X.copy()
    Yc = Y.copy()
    gamma = 1. / (2 * sigma**2)
    # Xc *= sqrt(2 * gamma)  # sklearn has k_rbf = exp(-gamma||x-y||^2) 
    Xc /= sigma
    Yc /= sigma

    D = get_D(d, n)
    b = lambda w: 1 - np.mean(np.power(w, 2)) # compensate for the first feature

    Z = np.vstack((Xc, Yc))

    Mz, w = mapping(Z, n)  # input scaled
    Mx, My = np.split(Mz, [X.shape[0]], 0)
    K = np.dot(Mx, My.T)
    K /= D #why we need this?
    K += b(w)
    return K


def mapping(Z, n, **kwargs):
    f = butt_quad_mapping
    non_linear = lambda mx: np.hstack((np.cos(mx), np.sin(mx)))
    even = True

    r = kwargs.get('r', None)
    b_params = kwargs.get('b_params', None)
    MZ, w = f(Z.T, n, r=r, b_params=b_params, even=even)
    MZ = MZ.T

    MZ = non_linear(MZ)
    ww = np.concatenate((w, w))
    MZ = MZ@np.diag(ww)
    # MZ = np.einsum('j,ij->ij', ww, MZ)

    return MZ, w
