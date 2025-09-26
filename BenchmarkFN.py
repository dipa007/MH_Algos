#######BENCHMARKING##############
def F1(x): return np.sum(x**2)
def F2(x): return np.sum(np.abs(x)) + np.prod(np.abs(x))
def F3(x):
    s = 0
    for i in range(len(x)): s += (np.sum(x[0:i+1]))**2
    return s
def F4(x): return np.max(np.abs(x))
def F5(x):
    dim = len(x)
    return np.sum(100 * (x[1:dim] - (x[0:dim-1]**2))**2 + (x[0:dim-1] - 1)**2)
def F6(x): return np.sum((np.floor(x + 0.5))**2)
def F7(x): return np.sum(np.arange(1, len(x)+1) * (x**4)) + np.random.uniform(0, 1)
def F8(x): return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
def F9(x): return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
def F10(x):
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e
def F11(x):
    return (1/4000) * np.sum(x**2) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1
def u_penalty_function(val, a, k=100, m=4):
    """
    The penalty function u() as defined in the paper for F12 and F13.
    This is a direct mathematical translation without custom precision handling.
    """
    if val > a:
        return k * ((val - a) ** m)
    elif val < -a:
        return k * ((-val - a) ** m)
    else:
        return 0.0

def F12(x):
    """F12 implementation using the original u() penalty function."""
    n = len(x)
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    penalty = np.sum([u_penalty_function(val, 10) for val in x])
    return (np.pi / n) * (term1 + term2 + term3) + penalty

def F13(x):
    """F13 implementation using the original u() penalty function."""
    n = len(x)
    term1 = 0.1 * np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = np.sum([u_penalty_function(val, 5) for val in x])
    return term1 + 0.1 * (term2 + term3) + penalty
def F14(x):
    a = np.array([[-32, -16, 0, 16, 32]*5, [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5])
    s = sum(1 / (j + 1 + (x[0] - a[0, j])**6 + (x[1] - a[1, j])**6) for j in range(25))
    return (1/500 + s)**-1
def F15(x):
    a = np.array([0.00030,0.00138,0.00138,0.00138,0.00138,0.00030,0.00138,0.00138,0.00138,0.00138,0.00030])
    b = np.array([-2.477,-2.477,-2.477,-2.477,-2.477,2.477,2.477,2.477,2.477,2.477,2.477])
    return np.sum((a - x[0]*(b**2 + b*x[1]) / (b**2 + b*x[2] + x[3]))**2)
def F16(x): return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
def F17(x): return (x[1]-(5.1/(4*np.pi**2))*x[0]**2+(5/np.pi)*x[0]-6)**2+10*(1-(1/(8*np.pi)))*np.cos(x[0])+10
def F18(x):
    return (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)) * \
           (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
def F19(x):
    a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    p = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    c = np.array([1, 1.2, 3, 3.2])
    return -np.sum(c * np.exp(-np.sum(a * (x - p)**2, axis=1)))
def F20(x):
    a = np.array([[10,3,17,3.5,1.7,8], [0.05,10,17,0.1,8,14], [3,3.5,1.7,10,17,8], [17,8,0.05,10,0.1,14]])
    p = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
                  [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    c = np.array([1, 1.2, 3, 3.2])
    return -np.sum(c * np.exp(-np.sum(a * (x - p)**2, axis=1)))
def _shekel(x, a, c, m): return sum(-1 / (np.sum((x - a[i])**2) + c[i]) for i in range(m))
def F21(x):
    a=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7]]); c=np.array([0.1,0.2,0.2,0.4,0.4])
    return _shekel(x,a,c,5)
def F22(x):
    a=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3]]); c=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3])
    return _shekel(x,a,c,7)
def F23(x):
    a=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]); c=np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    return _shekel(x,a,c,10)
     
