'''
TMA4212 Numerical solution of partial differential equations by difference methods.
Exercise 1.
'''
import numpy as np
import numpy.linalg as la
from numpy import inf
import matplotlib.pyplot as plt

# The following is some settings for the figures.
# This can be manipulated to get nice plots included in pdf-documents.
newparams = {'figure.figsize': (8.0, 5.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)


def tridiag(c, a, b, N):
    # Returns a tridiagonal matrix A=tridiag(c, a, b) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = c*np.diag(e[1:],-1)+a*np.diag(e)+b*np.diag(e[1:],1)
    return A

def bvp(f, alpha, beta, M=10):
    # Solve the BVP -u''(x)=f(x), u(0)=alpha, u(1)=beta
    # by a central difference scheme.
    h = 1/M
    Ah = tridiag(-1,2,-1,M-1)/h**2      # Set up the coefficient matrix
    x = np.linspace(0,1,M+1)    # gridpoints, including the boundary points
    xi = x[1:-1]             # inner gridpoints
    F = f(xi)                # evaluate f in the inner gridpoints
    F[0] = F[0]+alpha/h**2   # include the contribution from the boundaries
    F[-1] = F[-1]+beta/h**2

    # Solve the linear equation system
    Ui = la.solve(Ah, F)        # the solution in the inner gridpoints

    # Include the boundary points in the solution vector
    U = np.zeros(M+1)
    U[0] = alpha
    U[1:-1] = Ui
    U[-1] = beta
    return x, U

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Set up the problem to be solved.

    def f(x):
        y = np.pi**2*np.sin(x*np.pi)
        return y

    # Question 1 c) e(h) logplot
    def sol(x):
        y = np.sin(x*np.pi)
        return y


    alpha, beta = 0, 0         # boundary values

    # Solve the BVP
    x, U = bvp(f, alpha, beta, M=10)

    #And plot the solution
    plt.plot(x, U,'.-')
    plt.plot(x, sol(x), '.-')
    plt.xlabel('x')
    plt.ylabel('U')
    plt.title('Numerical solution of the model problem')
    plt.legend(['U'])
    plt.show()




    h_dat = []
    e_dat = []
    for i in range(2,10):
        M = 2**i
        # Append h data
        h_dat.append(np.log(1.0/float(M)))
        # Get solution
        x, U = bvp(f, alpha, beta, M)
        # Get error vector
        e_vec = sol(x)-U
        # Get 2 norm of the error
        e = np.linalg.norm(e_vec, 2)
        # Append e data
        e_dat.append(np.log(e))

    # Plot log(e) ifo log(h)
    plt.plot(h_dat,e_dat,'.-')
    plt.xlabel('log(h)')
    plt.ylabel('log(e)')
    plt.title('Error in function of M')
    plt.legend(['||e||_2'])
    plt.show()
