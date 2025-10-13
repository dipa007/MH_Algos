import numpy as np
import random
import math

LB = -100
UB = 100

def pressure_vessel(x, d):
    """
    Pressure Vessel Design Problem.
    Decision variables:
        x[0]: shell thickness (x1)
        x[1]: head thickness (x2)
        x[2]: inner radius (x3)
        x[3]: length of the cylindrical section (x4)
    d: dimension (should be 4)
    """
    # Unpack the decision variables
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    # Calculate the cost (objective function)
    cost = (0.6224 * x1 * x3 * x4 +
            1.7781 * x2 * (x3**2) +
            3.1661 * (x1**2) * x4 +
            19.84 * (x1**2) * x3)
    
    # Constraint functions (must be <= 0)
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -math.pi * (x3**2) * x4 - (4.0/3.0) * math.pi * (x3**3) + 1296000
    g4 = x4 - 240
    
    # Penalty for constraint violation
    penalty = 0.0
    penalty_factor = 1e6  # you may adjust this factor if needed

    if g1 > 0:
        penalty += penalty_factor * g1 ** 2
    if g2 > 0:
        penalty += penalty_factor * g2 ** 2
    if g3 > 0:
        penalty += penalty_factor * g3 ** 2
    if g4 > 0:
        penalty += penalty_factor * g4 ** 2

    return cost + penalty

def welded_beam(x, d):
    """
    Welded Beam Design Problem with extended constraints.
    
    Decision Variables:
      x[0] : weld thickness, x1
      x[1] : weld length,    x2
      x[2] : beam height,    x3
      x[3] : beam thickness, x4
    d : dimension (should be 4)
    
    The objective is to minimize:
      f(x) = 1.10471*x1^2*x2 + 0.04811*x3*x4*(14 + x2)
    
    Subject to the constraints (g_i(x) <= 0):
      g1(x) = tau(x) - TAUMAX
      g2(x) = sigma(x) - SIGMAX
      g3(x) = x1 - x4
      g4(x) = 0.10471*x1^2 + 0.04811*x3*x4*(14 + x2) - 5
      g5(x) = 0.125 - x1
      g6(x) = delta(x) - DELTMAX
      g7(x) = P - Pc(x)
    
    where the intermediate functions are defined as follows:
      P     = 6000           (applied tip load)
      E     = 30e6           (Young's modulus)
      G     = 12e6           (Shear modulus)
      L     = 14             (length of the beam)
      TAUMAX= 13600          (max allowed shear stress)
      SIGMAX= 30000          (max allowed bending stress)
      DELTMAX=0.25           (max allowed tip deflection)
      
      M(x)      = P * (L + x2/2)         (bending moment)
      R(x)      = sqrt((x2^2)/4 + ((x1 + x3)/2)^2)
      J(x)      = 2*sqrt(2)*x1*x2*((x2^2)/12 + ((x1+x3)/2)^2)   (polar moment of inertia)
      
      sigma(x)= (6*P*L)/(x4*x3^2)        (bending stress)
      delta(x)= (4*P*L^3)/(E*x4*x3^3)      (tip deflection)
      
      Pc(x)   = 4.013*E*sqrt((x3^2*x4^6)/36) * (1 - x3*sqrt(E/(4*G))/(2*L))/(L^2)   (buckling load)
      
      tau_p(x)= P/(sqrt(2)*x1*x2)
      tau_pp(x)= (M(x)*R(x))/J(x)
      tau(x)= sqrt( tau_p(x)^2 + (tau_p(x)*tau_pp(x)*x2)/R(x) + tau_pp(x)^2 )
      
    A penalty term is added for any constraint violation.
    """
    # Constants
    P = 6000.0
    E = 30e6
    G = 12e6
    L_val = 14.0      # Beam length
    PCONST = 1e6      # Penalty constant
    TAUMAX = 13600.0
    SIGMAX = 30000.0
    DELTMAX = 0.25

    # Unpack decision variables
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    # Objective function
    f_obj = 1.10471 * (x1**2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    
    # Intermediate functions
    M = P * (L_val + x2/2.0)  # Bending moment at the welding point
    R_val = math.sqrt((x2**2)/4.0 + ((x1 + x3)/2.0)**2)
    J = 2.0 * math.sqrt(2.0) * x1 * x2 * (((x2**2)/12.0) + ((x1 + x3)/2.0)**2)
    
    sigma_val = 504000 / (x4 * (x3**2))
    delta_val = 65856000 / (E * x4 * (x3**3))
    
    Pc = 4.013 * E * math.sqrt((x3**2 * x4**6) / 36.0) * (1 - x3 * math.sqrt(E / (4.0 * G)) / (2.0 * L_val)) / (L_val**2)
    
    tau_p = P / (math.sqrt(2.0) * x1 * x2)
    tau_pp = (M * R_val) / J
    tau_val = math.sqrt(tau_p**2 + (tau_p * tau_pp * x2) / R_val + tau_pp**2)
    
    # Constraint functions (all should be <= 0)
    g1 = tau_val - TAUMAX
    g2 = sigma_val - SIGMAX
    g3 = x1 - x4
    g4 = 0.10471 * x1**2 + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0
    g5 = 0.125 - x1
    g6 = delta_val - DELTMAX
    g7 = P - Pc

    # Helper function: returns the positive part of a value
    def pos(val):
        return val if val > 0 else 0.0

    # Penalty: square the violation of each constraint and sum them up
    penalty = PCONST * (pos(g1)**2 + pos(g2)**2 + pos(g3)**2 +
                        pos(g4)**2 + pos(g5)**2 + pos(g6)**2 + pos(g7)**2)
    
    return f_obj + penalty

def car_side_impact(x, d):
    """
    Car Side Impact Design Problem.
    """
    # Penalty constant
    PCONST = 1e6
    
    # Unpack decision variables
    z1, z2, z3, z4, z5, z6, z7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    
    # Calculate intermediate values (for clarity)
    V_mbp = 10.58 - 0.67275 * z2 - 0.674 * z1 * z2
    V_fd = 16.45 - 0.489 * z3 * z7 - 0.843 * z5 * z6
    
    # Objective functions
    f1 = 1.98 + 4.9*z1 + 6.67*z2 + 6.98*z3 + 4.01*z4 + 1.78*z5 + 0.00001*z6 + 2.73*z7
    f2 = 4.72 - 0.5*z4 - 0.19*z2*z3
    f3 = 0.5 * (V_mbp + V_fd)
    
    # Combine objectives with weights
    w1, w2, w3 = 0.4, 0.3, 0.3
    f_obj = w1*f1 + w2*f2 + w3*f3
    
    # Constraints (all should be <= 0)
    g1 = 1.16 - 0.3717*z2*z4 - 0.0092928*z3 - 1.0
    g2 = 0.261 - 0.0159*z1*z2 - 0.06486*z1 - 0.019*z2*z7 + 0.0144*z3*z5 + 0.0154464*z6 - 0.32
    g3 = 0.214 - 0.0587118*z1 + 0.018*z2**2 + 0.030408*z3 + 0.00817*z5 + 0.03099*z2*z6 - 0.018*z2*z7 - 0.00364*z5*z6 - 0.32
    g4 = 0.74 - 0.61*z2 + 0.227*z2**2 - 0.031296*z3 - 0.031872*z7 - 0.32
    g5 = 28.98 + 3.818*z3 - 4.2*z1*z2 + 1.27296*z6 - 2.68065*z7 - 32
    g6 = 33.86 + 2.95*z3 - 5.057*z1*z2 - 3.795*z2 - 3.4431*z7 + 1.45728 - 32
    g7 = 46.36 - 9.9*z2 - 4.4505*z1 - 32
    g8 = 4.72 - 0.5*z4 - 0.19*z2*z3 - 4.0
    g9 = V_mbp - 9.9
    g10 = V_fd - 15.7
    
    # Helper function for calculating penalties
    def pos(val):
        return val if val > 0 else 0.0
    
    # Calculate total penalty
    penalty = PCONST * (pos(g1)**2 + pos(g2)**2 + pos(g3)**2 + pos(g4)**2 + 
                        pos(g5)**2 + pos(g6)**2 + pos(g7)**2 + pos(g8)**2 +
                        pos(g9)**2 + pos(g10)**2)
    
    # Return objective plus penalty
    return f_obj + penalty

    
def speed_reducer(x, d):
    
    # Penalty constant (PCONST)
    PCONST = 1e6
    
    # Unpack decision variables
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    
    # Objective function
    obj = (0.7854 * x1 * (x2**2) * (3.3333 * (x3**2) + 14.9334 * x3 - 43.0934) 
            - 1.508 * x1 * (x6**2 + x7**2) 
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * (x6**2) + x5 * (x7**2)))
            
    # Constraint functions (should be <= 0)
    g1  = 27 / (x1 * (x2**2) * x3) - 1
    g2 = 397.5 / (x1 * (x2**2) * (x3**2))   - 1
    g3  = (1.93 * (x4**3)) / (x2 * x3 * (x6**4)) - 1    
    g4  = (1.93 * (x5**3)) / (x2 * x3 * (x7**4)) - 1
    g5  = (1 / (110 * (x6**3))) * math.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) - 1
    g6  = (1 / (85 * (x7**3))) * math.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) - 1
    g7  = (x2 * x3 / 40) - 1
    g8  = (5 * x2 / x1) - 1
    g9  = (x1 / (12 * x2)) - 1
    g10 = ((1.5 * x6 + 1.9) / x4) - 1
    g11 = ((1.1 * x7 + 1.9) / x5) - 1
    
    # Helper: positive part function
    def pos(val):
        return val if val > 0 else 0.0
    
    # Sum the squares of all constraint violations
    penalty = PCONST * ( pos(g1)**2 + pos(g2)**2 + pos(g3)**2 + pos(g4)**2 +
                         pos(g5)**2 + pos(g6)**2 + pos(g7)**2 + pos(g8)**2 +
                         pos(g9)**2 + pos(g10)**2 + pos(g11)**2 )
    
    return obj + penalty

def tension_spring(x, d):
    """
    Tension Spring Design Problem.
    
    Decision Variables (x is a 3-element vector):
      x[0] : wire diameter (x1)
      x[1] : mean coil diameter (x2)
      x[2] : number of active coils (x3)
    
    Objective Function:
      f(x) = (x3 + 2) * x2 * (x1^2)
    
    Constraint Functions (should be <= 0):
      g1(x) = 1 - ( x2^3 * x3 )/( 71785 * x1^4 )
      g2(x) = ( (4*x2^2 - x1*x2) / (12566 * (x2*x1^3 - x1^4)) ) + ( (1/5108)*x1^2 )
      g3(x) = 1 - ( 140.45*x1 )/( x2^2 * x3 )
      g4(x) = ((x1 + x2) / 1.5) - 1
    
    A penalty term is added for any constraint violation.
    
    Parameters:
      x : array_like
          A 3-element vector of decision variables [x1, x2, x3].
      d : int
          Dimension of the problem (should be 3).
    
    Returns:
      f_penalized : float
          The objective function value plus penalty for any constraint violation.
    """
    # Penalty constant
    PCONST = 1e6
    
    # Unpack decision variables
    x1, x2, x3 = x[0], x[1], x[2]
    
    # Objective function: f(x) = (x3 + 2)*x2*(x1^2)
    f_obj = (x3 + 2) * x2 * (x1**2)
    
    # Constraint functions
    # g1: 1 - (x2^3*x3)/(7178*x1^4) <= 0
    g1 = 1 - ((x2**3 * x3) / (71785 * (x1**4)))
    
    # g2: ((4*x2^2 - x1*x2)/(12566*(x2*x1^3))) + ((1/5108)*x1^2) - 1 <= 0
    g2 = (4*x[1]**2 - x[0]*x[1])/(12566 * (x[1]*x[0]**3 - x[0]**4)) + 1 / (5108 * x[0]**2) - 1
    
    # g3: 1 - (140.45*x1)/(x2^2*x3) <= 0
    g3 = 1 - ((140.45 * x1) / ((x2**2) * x3))
    
    # g4: ((x1 + x2)/1.5) - 1 <= 0
    g4 = ((x1 + x2) / 1.5) - 1
    
    # Helper: returns the positive part (if violation exists, else zero)
    def pos(val):
        return val if val > 0 else 0.0
    
    # Total penalty: square each violation and multiply by PCONST
    penalty = PCONST * ( pos(g1)**2 + pos(g2)**2 + pos(g3)**2 + pos(g4)**2 )
    
    # Penalized objective
    return f_obj + penalty

def gear_train(x, d):
    """
    Gear Train Design Problem.
    
    Decision Variables:
      x[0]: number of teeth on gear 1
      x[1]: number of teeth on gear 2
      x[2]: number of teeth on gear 3
      x[3]: number of teeth on gear 4
      
    Objective Function:
      f(x) = (1/6.931 - (x1*x2)/(x3*x4))^2
      
    Bounds (typical):
      12 <= x[i] <= 60, for i = 0,1,2,3.
      
    Note: In practice the variables are integers, but here we use continuous values.
    """
    # Desired gear ratio (approximately 1/6.931)
    ratio_desired = 1 / 6.931
    # Compute the objective value
    f_obj = (ratio_desired - (x[0] * x[1]) / (x[2] * x[3]))**2
    return f_obj

def cantilever_beam(x, d):
    """
    Cantilever Beam Design Problem.
    
    This is a 5-dimensional problem representing a cantilever beam made of 
    five hollow square sections.
    
    Decision Variables:
      x[0]-x[4]: Dimensions of the five hollow square sections (x1, x2, x3, x4, x5)
      
    Objective Function:
      Minimize the total weight: 0.06224 * (x1 + x2 + x3 + x4 + x5)
      
    Constraints:
      g1(x): (61/x1**3) + (27/x2**3) + (19/x3**3) + (7/x4**3) + (1/x5**3) - 1 <= 0
    """

    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]

    # Objective function: minimize total weight
    f_obj = 0.06224 * (x1 + x2 + x3 + x4 + x5)
    
    # Constraint g1: load capacity constraint
    g1 = (61/x1**3)+ (27/x2**3) + (19/x3**3) + (7/x4**3) + (1/x5**3) - 1

    # # Death penalty approach - if any constraint is violated, return infinity
    # if g1 > 0:
    #     return float('inf')  # Complete rejection of infeasible solutions
    
    # return f_obj  # Return actual objective value for feasible solutions

    # Penalty handling
    PCONST = 1e6
    def pos(val):
        return val if val > 0 else 0.0
    penalty = PCONST * (pos(g1)**2)
    
    return f_obj + penalty


def rolling_element_bearing_design(x, d):
    """
    Rolling Element Bearing Design Problem - MAXIMIZATION problem.
    The goal is to maximize the dynamic loading carrying capacity.
    
    Decision Variables:
      x[0] : Dm   (pitch/mean diameter)
      x[1] : Db   (ball diameter)
      x[2] : Z    (number of balls)
      x[3] : fi   (inner raceway curvature coefficient)
      x[4] : f0   (outer raceway curvature coefficient)
      x[5] : Kdmin
      x[6] : Kdmax
      x[7] : theta
      x[8] : e    (eccentricity)
      x[9] : C    (contact factor)
    """
    # Problem constants
    D = 160.0       # Outer diameter
    d_const = 90.0  # Inner diameter
    Bw = 30.0       # Width
    eps = 1e-10     # Small value for numerical stability
    
    # Unpack decision variables (ensure Z is integer)
    Dm, Db, Z_float, fi, f0, Kdmin, Kdmax, theta, e, C = x
    Z = max(4, min(50, int(round(Z_float))))  # Ensure Z is integer within [4, 50]
    
    # Calculate intermediate values
    T = D - d_const - 2 * Db
    gamma = Db / Dm  # Ratio of ball diameter to pitch diameter
    
    # Calculate theta0 (geometric parameter)
    try:
        xx = ((D - d_const)/2 - 3*(T/4))**2 + (D/2 - T/4 - Db)**2 - ((d_const/2) + T/4)**2
        yy = 2 * ((D - d_const)/2 - 3*T/4) * (D/2 - T/4 - Db)
        
        if abs(yy) < eps:
            theta0 = math.pi
        else:
            cos_arg = max(min(xx / yy, 1.0), -1.0)
            theta0 = 2 * math.pi - math.acos(cos_arg)
    except:
        theta0 = math.pi
    
    # Calculate dynamic capacity factor (fc)
    try:
        # Term calculations with safety checks
        term1 = max(0, (1 - gamma) / (1 + gamma))**1.72
        term2 = max(0, (2 * f0 - 1) / max(2 * Z - 1, eps))**0.41
        term3 = (fi / max(f0, eps)) * term2
        term3 = max(0, term3)**(10.0/3)
        
        t1 = 37.91 * max(1 + 1.04 * term1 * term3, eps)**(-0.3)
        
        # Calculate fc
        term4 = gamma**0.3 * (1 - gamma)**1.39
        term5 = max(1 + gamma, eps)**(1/3)
        term6 = (2 * Z / max(2 * Z - 1, eps))**0.41
        
        fc = t1 * (term4 / term5) * term6
        
        # Calculate objective function (Cd)
        if Db <= 25.4:
            Cd = fc * (Z**(2/3)) * (Db**1.8)
        else:
            Cd = 3.647 * fc * (Z**(2/3)) * (Db**1.4)
    except:
        Cd = 0
    
    # Calculate constraints
    constraints = [0] * 9
    try:
        sin_term = max(math.sin(Db/(Dm + eps)), eps)
        sin_inv_term = math.asin(Db/(Dm + eps))

        constraints[0] = theta0 / (2*sin_inv_term) - Z + 1              # g1
        constraints[1] = -(2*Db - Kdmin*(D-d_const))                    # g2
        constraints[2] = -(Kdmax*(D-d_const) - 2*Db)                    # g3
        constraints[3] = Db - C*Bw                                      # g4
        constraints[4] = -(Dm - 0.5*(D+d_const))                        # g5
        constraints[5] = -((0.5+e)*(D+d_const) - Dm)                    # g6
        constraints[6] = -(0.5*(D-Dm-Db) - theta*Db)                    # g7
        constraints[7] = 0.515 - fi                                     # g8
        constraints[8] = 0.515 - f0                                     # g9
    except:
        constraints = [1] * 9
    
    # Apply penalties for constraint violations
    penalty = 0.0
    for g in constraints:
        if g > 0:
            penalty += 1e6 * g**2
    
    # Return negative objective (for maximization) plus penalty
    return -Cd + penalty

def i_beam_design(x, d):
    """
    I-Beam Design Problem based on the equations in the provided image.

    Decision Variables:
      x[0] : v  (width of the flanges)
      x[1] : u  (height of the component)
      x[2] : wt (thickness of the web)
      x[3] : ft (thickness of the flanges)

    Objective Function:
      Minimize the fitness function as defined in the image.

    Constraints:
      g1(x) : Constraint y1 from the image
      g2(x) : Constraint y2 from the image
    """
    # Penalty constant
    PCONST = 1e6  # Increased penalty for strict enforcement

    # Unpack decision variables
    v, u, wt, ft = x[0], x[1], x[2], x[3]

    # Problem constants (assuming x1 and x2 in the image correspond to u and v respectively for now)
    x1 = u
    x2 = v

    # Calculate objective function (fitness)
    term1 = (wt * (v - 2 * ft)**3) / 12
    term2 = (u * ft**3) / 6
    term3 = 2 * u * ft * ((v - ft) / 2)**2
    objective = 5000 / (term1 + term2 + term3) # if (term1 + term2 + term3) != 0 else float('inf')

    # Calculate constraints (all should be <= 0)
    g1 = (2 * u * ft + wt * (v - 2 * ft)**3) - 300
    term_y2_den1 = (wt * (v - 2 * ft)**3)
    term_y2_den2 = 2 * u * ft * (4 * ft**2 + 3 * v * (v - ft))
    term_y2_den3 = (2 * ft * u**3) / 3
    denominator1 = term_y2_den1 + term_y2_den2 # + term_y2_den3

    term_y2_den4 = (v - 2 * ft) * wt**3
    term_y2_den5 = 2 * ft * u**3
    denominator2 = term_y2_den4 + term_y2_den5

    term_y2_num1 = 180000 * x1
    term_y2_num2 = 15000 * x2

    val1 = term_y2_num1 / denominator1 # if denominator1 != 0 else float('inf')
    val2 = term_y2_num2 / denominator2 # if denominator2 != 0 else float('inf')

    g2 = (val1 + val2) - 6

    # Helper function: returns the positive part of a value
    def pos(val):
        return val if val < 0 else 0.0

    # Calculate penalty
    penalty = PCONST * (pos(g1)**2 + pos(g2)**2)

    return objective + penalty


def fifteen_bar_truss(x, d):
    """
    15-Bar Truss Design Problem.
    
    Decision Variables:
      x[0]-x[14]: Cross-sectional areas for each of the 15 bars
    
    Objective Function:
      Minimize the weight of the structure
    
    Constraints:
      - Stress constraints for each member
      - Deflection constraints at specific nodes
    """
    # Problem constants
    E = 10000  # Young's modulus (ksi)
    rho = 0.1  # Material density (lb/in³)
    P = 10000  # Applied load (lb)
    sigma_max = 25  # Maximum allowable stress (ksi)
    delta_max = 2.0  # Maximum allowable deflection (in)
    
    # Node coordinates (inches)
    nodes = np.array([
        [0, 0], [0, 120], [120, 0], [120, 120], [240, 0], [240, 120]
    ])
    
    # Member connectivity (0-indexed)
    members = np.array([
        [0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2],
        [2, 4], [4, 5], [3, 5], [2, 5], [3, 4], [1, 4], [1, 5], [0, 5]
    ])
    
    # Member lengths
    lengths = np.zeros(15)
    for i in range(15):
        node1, node2 = members[i]
        lengths[i] = np.linalg.norm(nodes[node1] - nodes[node2])
    
    # Calculate weight (objective function)
    weight = sum(x[i] * lengths[i] * rho for i in range(15))
    
    # Truss analysis would be done here to get stresses and deflections
    # This is a simplified version without actual FEA analysis
    
    # Simulate stress constraints (would normally come from FEA)
    stresses = np.zeros(15)
    for i in range(15):
        # Simplified stress calculation
        stresses[i] = P / x[i] * np.random.uniform(0.5, 1.5)
    
    # Simulate nodal deflections (would normally come from FEA)
    deflections = np.zeros((6, 2))
    for i in range(6):
        # Simplified deflection calculation
        deflections[i] = np.random.uniform(0, delta_max * 0.8, 2)
    
    # Calculate constraint violations
    stress_violations = np.maximum(0, np.abs(stresses) - sigma_max)
    deflection_violations = np.maximum(0, np.linalg.norm(deflections, axis=1) - delta_max)
    
    # Sum up penalties
    penalty = 1e6 * (sum(stress_violations**2) + sum(deflection_violations**2))
    
    return weight + penalty

def twenty_five_bar_truss(x, d):
    """
    25-Bar Truss Design Problem.
    
    Decision Variables:
      x[0]-x[7]: Cross-sectional areas for groups of bars (8 groups total)
    
    Objective Function:
      Minimize the weight of the structure
    
    Constraints:
      - Stress constraints for each member
      - Displacement constraints at specific nodes
    """
    # Problem constants
    E = 10000  # Young's modulus (ksi)
    rho = 0.1  # Material density (lb/in³)
    sigma_max = 40  # Maximum allowable stress (ksi)
    delta_max = 0.35  # Maximum allowable deflection (in)
    
    # Node coordinates (inches)
    nodes = np.array([
        [0, 0, 0], [0, 0, 75], [37.5, 0, 100], [75, 0, 75], [75, 0, 0],
        [37.5, 37.5, 100], [37.5, -37.5, 100], [37.5, 37.5, 0], [37.5, -37.5, 0]
    ])
    
    # Member connectivity (0-indexed) with group assignments
    members = [
        # Group 1 (A1)
        [[0, 1]],
        # Group 2 (A2)
        [[1, 2], [1, 3], [1, 5], [1, 6]],
        # Group 3 (A3)
        [[2, 3], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]],
        # Group 4 (A4)
        [[0, 4], [4, 8]],
        # Group 5 (A5)
        [[0, 7], [0, 8], [4, 7], [4, 8]],
        # Group 6 (A6)
        [[5, 6]],
        # Group 7 (A7)
        [[2, 5], [3, 6]],
        # Group 8 (A8)
        [[5, 8], [6, 7], [6, 8], [7, 8]]
    ]
    
    # Calculate member lengths and initialize arrays
    all_members_flat = [item for sublist in members for item in sublist]
    lengths = np.zeros(len(all_members_flat))
    areas = np.zeros(len(all_members_flat))
    
    # Assign cross-sectional areas based on group assignments
    member_index = 0
    for i, group in enumerate(members):
        for _ in group:
            areas[member_index] = x[i]
            member_index += 1
    
    # Calculate lengths
    for i, mem in enumerate(all_members_flat):
        node1, node2 = mem
        lengths[i] = np.linalg.norm(nodes[node1] - nodes[node2])
    
    # Calculate weight
    weight = sum(areas[i] * lengths[i] * rho for i in range(len(lengths)))
    
    # In a real implementation, FEA would be used to calculate stresses and deflections
    # Here we use simplified calculations
    
    # Simulate stress constraints
    stresses = np.zeros(len(all_members_flat))
    for i in range(len(all_members_flat)):
        # Simplified stress calculation
        stresses[i] = np.random.uniform(20, 60) * (1000 / areas[i])
    
    # Simulate nodal deflections
    deflections = np.zeros((len(nodes), 3))
    for i in range(len(nodes)):
        # Simple deflection model
        deflections[i] = np.random.uniform(-delta_max, delta_max, 3)
    
    # Calculate constraint violations
    stress_violations = np.maximum(0, np.abs(stresses) - sigma_max)
    deflection_violations = np.zeros(len(nodes))
    for i in range(len(nodes)):
        deflection_violations[i] = max(0, np.linalg.norm(deflections[i]) - delta_max)
    
    # Sum up penalties
    penalty = 1e6 * (sum(stress_violations**2) + sum(deflection_violations**2))
    
    return weight + penalty

def fifty_two_bar_truss(x, d):
    """
    52-Bar Truss Design Problem.
    
    Decision Variables:
      x[0]-x[11]: Cross-sectional areas for groups of bars (12 groups total)
    
    Objective Function:
      Minimize the weight of the structure
    
    Constraints:
      - Stress constraints for each member
      - Displacement constraints at specific nodes
    """
    # Problem constants
    E = 2.07e7  # Young's modulus (kg/cm²)
    rho = 7860e-6  # Material density (kg/cm³)
    sigma_max = 20000  # Maximum allowable stress (kg/cm²)
    delta_max = 2.5  # Maximum allowable deflection (cm)
    
    # Define node coordinates (typical 52-bar tower truss)
    height = 1000  # cm
    width = 200  # cm
    levels = 4
    level_height = height / levels
    
    # Generate nodes
    nodes = []
    for i in range(levels + 1):  # 5 levels (0 to 4)
        z = i * level_height
        if i == 0:  # Base level has 4 nodes
            nodes.extend([
                [width/2, width/2, z],
                [width/2, -width/2, z],
                [-width/2, -width/2, z],
                [-width/2, width/2, z]
            ])
        else:  # Upper levels have 4 nodes each
            # Width decreases with height
            level_width = width * (1 - 0.1 * i)
            nodes.extend([
                [level_width/2, level_width/2, z],
                [level_width/2, -level_width/2, z],
                [-level_width/2, -level_width/2, z],
                [-level_width/2, level_width/2, z]
            ])
    nodes = np.array(nodes)
    
    # Define member grouping (12 groups)
    # In a real implementation, this would be a more detailed set of member definitions
    groups = [
        # Group 1: Vertical members in the first level
        [[0, 4], [1, 5], [2, 6], [3, 7]],
        # Group 2: Diagonal members in the first level
        [[0, 5], [1, 6], [2, 7], [3, 4]],
        # Group 3: Horizontal members in the first level
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        # Group 4: Vertical members in the second level
        [[4, 8], [5, 9], [6, 10], [7, 11]],
        # Group 5: Diagonal members in the second level
        [[4, 9], [5, 10], [6, 11], [7, 8]],
        # Group 6: Horizontal members in the second level
        [[4, 5], [5, 6], [6, 7], [7, 4]],
        # Group 7: Vertical members in the third level
        [[8, 12], [9, 13], [10, 14], [11, 15]],
        # Group 8: Diagonal members in the third level
        [[8, 13], [9, 14], [10, 15], [11, 12]],
        # Group 9: Horizontal members in the third level
        [[8, 9], [9, 10], [10, 11], [11, 8]],
        # Group 10: Vertical members in the fourth level
        [[12, 16], [13, 17], [14, 18], [15, 19]],
        # Group 11: Diagonal members in the fourth level
        [[12, 17], [13, 18], [14, 19], [15, 16]],
        # Group 12: Horizontal members in the fourth level
        [[12, 13], [13, 14], [14, 15], [15, 12]]
    ]
    
    # Calculate member lengths and initialize arrays
    all_members_flat = [item for sublist in groups for item in sublist]
    lengths = np.zeros(len(all_members_flat))
    areas = np.zeros(len(all_members_flat))
    
    # Assign cross-sectional areas based on group assignments
    member_index = 0
    for i, group in enumerate(groups):
        for _ in group:
            areas[member_index] = x[i]
            member_index += 1
    
    # Calculate lengths
    for i, mem in enumerate(all_members_flat):
        node1, node2 = mem
        lengths[i] = np.linalg.norm(nodes[node1] - nodes[node2])
    
    # Calculate weight (objective function)
    weight = sum(areas[i] * lengths[i] * rho for i in range(len(lengths)))
    
    # In a real implementation, FEA would be used to calculate stresses and deflections
    # Here we use simplified calculations
    
    # Simulate stress constraints
    stresses = np.zeros(len(all_members_flat))
    for i in range(len(all_members_flat)):
        # Simplified stress calculation
        if areas[i] < 1e-6:  # Avoid division by zero
            stresses[i] = sigma_max * 2  # Guaranteed violation
        else:
            stresses[i] = np.random.uniform(10000, 30000) * (1 / areas[i])
    
    # Simulate nodal deflections
    deflections = np.zeros((len(nodes), 3))
    for i in range(len(nodes)):
        if i >= 4:  # Only upper nodes have deflections
            # Simple deflection model
            deflections[i] = np.random.uniform(-delta_max, delta_max, 3)
    
    # Calculate constraint violations
    stress_violations = np.maximum(0, np.abs(stresses) - sigma_max)
    deflection_violations = np.zeros(len(nodes))
    for i in range(len(nodes)):
        deflection_violations[i] = max(0, np.linalg.norm(deflections[i]) - delta_max)
    
    # Sum up penalties
    penalty = 1e6 * (sum(stress_violations**2) + sum(deflection_violations**2))
    
    return weight + penalty


def Get_Functions_details(F):

    if F == 'REB':  # Rolling Element Bearing design problem
        fobj = rolling_element_bearing_design
        dim = 10  # 10 decision variables
        lb = np.array([125., 10.5, 4, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6])
        ub = np.array([150., 31.5, 50, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85])
        # Wrapper function to round Z to integer:
        def fobj_wrapper(x, d):
            x = x.copy()
            x[2] = int(round(x[2]))
            return fobj(x, d)
        return lb, ub, dim, fobj_wrapper
    
    elif F == 'IBD':  # IBD for I-Beam Design
        fobj = i_beam_design
        dim = 4  # 4 decision variables
        # Define the bounds for the I-beam design problem
        # [h, b, tw, tf]
        lb = np.array([10.0, 10.0, 0.9, 0.9])
        ub = np.array([80.0, 50.0, 5.0, 5.0])
    
    elif F == 'CSI':  # Car Side Impact
        fobj = car_side_impact
        dim = 7  # 7 decision variables
        
        # Define the bounds as per the constraints
        lb = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4])
        ub = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2])
        
    elif F == '15BT':  # 15-Bar Truss
        fobj = fifteen_bar_truss
        dim = 15  # 15 decision variables (one for each bar area)
        lb = np.ones(dim) * 0.1  # Minimum area of 0.1 in²
        ub = np.ones(dim) * 35.0  # Maximum area of 35.0 in²
        
    elif F == '25BT':  # 25-Bar Truss
        fobj = twenty_five_bar_truss
        dim = 8  # 8 decision variables (one for each group)
        lb = np.ones(dim) * 0.1  # Minimum area of 0.1 in²
        ub = np.ones(dim) * 35.0  # Maximum area of 35.0 in²
        
    elif F == '52BT':  # 52-Bar Truss
        fobj = fifty_two_bar_truss
        dim = 12  # 12 decision variables (one for each group)
        lb = np.ones(dim) * 1.0  # Minimum area of 1.0 cm²
        ub = np.ones(dim) * 200.0  # Maximum area of 200.0 cm²

    elif F == 'CB':  # CB for Cantilever Beam
        fobj = cantilever_beam
        dim = 5  # Five sections
        # Define the bounds for the cantilever beam problem
        lb = np.ones(dim) * 0.01
        ub = np.ones(dim) * 100.0


    elif F == 'GT':  # GT for Gear Train
        fobj = gear_train
        dim = 4
        # Define the bounds for the gear train problem
        lb = np.array([12, 12, 12, 12])
        ub = np.array([60, 60, 60, 60])


    elif F == 'TS':  # TS for Tension Spring
        fobj = tension_spring
        dim = 3  # There are 3 decision variables
        # Define lower and upper bounds (typical values for the tension spring problem):
        lb = np.array([0.05, 0.25, 2.0])    # Lower bounds for [x1, x2, x3]
        ub = np.array([2.0, 1.3, 15.0])      # Upper bounds for [x1, x2, x3]

    elif F == 'SR':  # SR for Speed Reducer
        fobj = speed_reducer
        dim = 7  # The problem has 7 decision variables
        # Define variable bounds as NumPy arrays.
        # (The commonly used bounds for this problem are provided in many references;
        # you may adjust these bounds as needed.)
        lb = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
        ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])

    elif F == 'WB':  # WB for Welded Beam
        fobj = welded_beam
        dim = 4  # There are 4 design variables
        # Define the bounds as NumPy arrays for proper broadcasting
        lb = np.array([0.125, 0.1, 0.1, 0.1])
        ub = np.array([2.0, 10.0, 10.0, 2.0])


    elif F == 'PV':
        fobj = pressure_vessel
        dim = 4  # there are 4 design variables
        # Use vector bounds for the pressure vessel problem:
        lb = np.array([0.0625, 0.0625, 10, 10])
        ub = np.array([6.1875, 6.1875, 200, 200])

    
    # Get function details based on function name
    elif F == 'F1':
        fobj = f1
        lb = LB
        ub = UB
        dim = 30
    elif F == 'F2':
        fobj = f2
        lb = -10
        ub = 10
        dim = 30
    elif F == 'F3':
        fobj = f3
        lb = LB
        ub = UB
        dim = 30
    elif F == 'F4':
        fobj = f4
        lb = LB
        ub = UB
        dim = 30
    elif F == 'F5':
        fobj = f5
        lb = -5
        ub = 10
        dim = 30
    elif F == 'F6':
        fobj = f6
        lb = LB
        ub = UB
        dim = 30
    elif F == 'F7':
        fobj = f7
        lb = -1.28
        ub = 1.28
        dim = 30
    elif F == 'F8':
        fobj = f8
        lb = -500
        ub = 500
        dim = 30
    elif F == 'F9':
        fobj = f9
        lb = -5.12
        ub = 5.12
        dim = 30
    elif F == 'F10':
        fobj = f10
        lb = -32
        ub = 32
        dim = 30

    elif F == 'F11':
        fobj = f11
        lb = -600
        ub = 600
        dim = 30
    
    elif F == 'F12':
        fobj = f12
        lb = -50
        ub = 50
        dim = 30
    elif F == 'F13':  
        fobj = f12
        lb = -50
        ub = 50
        dim = 30
    elif F == 'F14':
        fobj = f14
        lb = -65
        ub = 65
        dim = 2
    elif F == 'F15':
        fobj = f15
        lb = -5
        ub = 5
        dim = 4
    elif F == 'F16':
        fobj = f16
        lb = -5
        ub = 5
        dim = 2
    elif F == 'F17':
        fobj = f17
        lb = -5
        ub = 5
        dim = 2
    elif F == 'F18':
        fobj = f18
        lb = -2
        ub = 2
        dim = 2
    elif F == 'F19':
        fobj = f19
        lb = 0
        ub = 1
        dim = 3
    elif F == 'F20':
        fobj = f20
        lb = 0
        ub = 1
        dim = 6
    elif F == 'F21':
        fobj = f21
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F22':
        fobj = f22
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F23':
        fobj = f23
        lb = 0
        ub = 10
        dim = 4
    
    
    else:
        raise ValueError("Unsupported function name")
    
    return lb, ub, dim, fobj


# Functions
def f1(x, d):
    f = 0
    for i in range(d):
        f += x[i] ** 2
    return f

def f2(x,d):
    s = 0
    p = 1
    for i in range(d):
        s += abs(x[i])
    for i in range(d):
        p *= abs(x[i])
    f = s + p
    return f

def f3(x,d):
    f = 0
    for i in range(1,d+1):
        s = 0
        for j in range(i):
            s += x[j]
        f += s**2
    return f

def f4(x,d):
    f = float("-inf")
    for i in range(d):
        if abs(x[i]) > f:
            f = abs(x[i])
    return f

def f5(x,d):
    f = 0
    for i in range(d-1):
        f += 100*((x[i+1] - x[i]**2)**2) + (x[i] - 1)**2
    return f 

def f6(x,d):
    f = 0
    for i in range(d):
        f += abs(x[i] + 0.5) ** 2     
    return f

def f7(x,d):
    f = 0
    for i in range(d):
        f += (i+1.0) * (x[i] ** 4.0)   
    return f + np.random.uniform(0,1)
    
def f8(x,d):
    f = 0
    for i in range(d):
        f += -(x[i] * math.sin(math.sqrt(abs(x[i]))))
    return f

def f9(x,d):
    f = 0
    p = math.pi
    for i in range(d):
        f += ((x[i]**2) - (10 * math.cos(2*p*x[i])) + 10)
    return f

def f10(x,d):
    p = math.pi
    s = 0
    n = 0
    for i in range(d):
        s += x[i]**2
    for i in range(d):
        n += math.cos(2*p*x[i])
    f = (-20 * np.exp(-0.2 * math.sqrt((1/d) * s))) - np.exp((1/d) * n) + 20 + np.e
    return f

def f11(x, d):
    p=1
    s = 0
    for i in range(d):
        p = p * math.cos(x[i]/math.sqrt(i+1))
        s += x[i]**2
    f =  (s/4000) - p + 1.0
    return f

def f12(n,d):
    def u(s,a,k,m):
        if s > a:
            return (k*((s-a)**m))
        elif s < -a:
            return (k*((-s-a)**m))
        else:
            return 0
    def y(s):
        return (1 + 0.25*(s+1))
    a = 10
    k = 100
    m = 4
    ui = 0
    xi = 0
    p = math.pi
    for i in range(d):
        ui += u(n[i],a,k,m)
    for i in range(d-1):
        xi += ((y(n[i])-1)**2) * (1 + 10*(math.sin(p*y(n[i+1]))**2)) 
    f = (p/(d+1)) * ( 10*(math.sin(p*y(n[0]))**2) + xi + (y(n[-1]) - 1)**2) + ui
    return f



def f13(n,d):
    def u(s,a,k,m):
        if s > a:
            return (k*((s-a)**m))
        elif s < -a:
            return (k*((-s-a)**m))
        else:
            return 0
    a = 5
    k = 100
    m = 4
    ui = 0
    xi = 0
    p = math.pi
    for i in range(d):
        ui += u(n[i],a,k,m)
    for i in range(d-1):
        xi += ((n[i]-1)**2) * (1 + math.sin(3*p*n[i+1])**2) 
    f = 0.1 * ( (math.sin(3*p*n[0])**2) + xi + (((n[-1] - 1)**2) * (1 + math.sin(2*p*n[-1])**2)) ) + ui
    return f

def f14(x, d):
    f = 0
    a = [[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],
         [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]]
    for j in range(1,26):
        s = 0
        for i in range(1,3):
            s += ((x[i-1]-a[i-1][j-1])**6)
        f += 1/(j+s)
    f = (1/((1/500) + f))
    return f

# Kowalik
def f15(m,d):  
    aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246]
    bK=[.25,.5,1,2,4,6,8,10,12,14,16]
    aK=np.asarray(aK)
    bK=np.asarray(bK)
    bK = 1/bK
    fit=np.sum((aK-((m[0]*(bK**2+m[1]*bK))/(bK**2+m[2]*bK+m[3])))**2)
    return fit

# Six Hump Camel function
def f16(x, d):
    f = ((4 - 2.1*(x[0]**2) + (x[0]**4)/3)*(x[0]**2)) + x[0]*x[1] + ((-4 + 4*(x[1]**2))*(x[1]**2))
    return f
# range = [-5, 5], d=2, value = −1.0316, n= 100, iter = 5000, p=20, I=C= 0.1, y1=y2=1


def f17(x, d):
    p = math.pi
    f = ((x[1] - (5.1/(4*(p**2)))*(x[0]**2) + ((5/p)*x[0]) - 6)**2) + (10*(1-(1/(8*p)))*math.cos(x[0])) + 10
    return f
# range = [-5,5], d = 2, value = 0.398


# Goldstein Price Function
def f18(x, d):
    s = 30 + (((2*x[0]) - (3*x[1]))**2) * (18-(32*x[0]) + (12*(x[0]**2)) + (48*x[1]) - (36*x[0]*x[1]) + (27*(x[1]**2)))
    first = (1 + ((x[0]+x[1]+1)**2) * (19-(14*x[0])+(3*(x[0]**2))-(14*x[1])+(6*x[0]*x[1])+ (3*(x[1]**2))))
    f = first * s
    return f
# range = [-2,2], d = 2, value = 3


# hartmann function 3
def f19(x, d):
    A = np.array([[ 3. , 10. , 30. ], [ 0.1, 10. , 35. ], [ 3. , 10. , 30. ], [ 0.1, 10. , 35. ]])
    P = np.array([[ 0.3689, 0.117 , 0.2673], [ 0.4699, 0.4387, 0.7470 ], [ 0.1091 , 0.8732, 0.5547], [ 0.0381, 0.5743, 0.8828]])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    f = 0
    for i in range(4):
        s = 0
        for j in range(3):
            s += (A[i][j] * (x[j] - P[i][j])**2)
        f += (alpha[i] * np.exp(-s))
    return -f
# range = [0,1], d=3, value = -3.86278, n= 100, iter = 5000, p=20, I=C= 0.1, y1=y2=1


# hartmann function 6
def f20(x, d):
    A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]])
    P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    alpha = alpha.transpose()
    f = 0
    for i in range(4):
        s = 0
        for j in range(6):
            s += (A[i][j] * ((x[j] - P[i][j])**2))
        f += (alpha[i] * np.exp(-s))
    return -f
# range = [0,1], d=6, value = -3.32278, n= 100, iter = 5000, p=20, I=C= 0.1, y1=y2=1


# shekel 5
def f21(x,d):
    a = [[4,4,4,4],
        [1,1,1,1],
        [8,8,8,8],
        [6,6,6,6],
        [3,7,3,7]]
    c = [0.1,0.2,0.2,0.4,0.4]
    f = 0
    s = 0
    for i in range(5):
        for j in range(d):
            s += ((x[j] - a[i][j])**2)
        f += (1/(s + c[i])) 
    return -f
#range = [0,10], d = 4, value = -10.1532


# shekel 7
def f22(x,d):
    a = [[4,4,4,4],
        [1,1,1,1],
        [8,8,8,8],
        [6,6,6,6],
        [3,7,3,7],
        [2,9,2,9],
        [5,5,3,3]]
    c = [0.1,0.2,0.2,0.4,0.4,0.6,0.3]
    f = 0
    s = 0
    for i in range(7):
        for j in range(d):
            s += ((x[j] - a[i][j])**2)
        f += (1/(s + c[i])) 
    return -f
#range = [0,10], d = 4, value = -10.4028

# shekel 10
def f23(x,d):
    a = [[4,4,4,4],
        [1,1,1,1],
        [8,8,8,8],
        [6,6,6,6],
        [3,7,3,7],
        [2,9,2,9],
        [5,5,3,3],
        [8,1,8,1],
        [6,2,6,2],
        [7,3.6,7,3.6]]
    c = [0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    f = 0
    s = 0
    for i in range(10):
        for j in range(d):
            s += ((x[j] - a[i][j])**2)
        f += (1/(s + c[i])) 
    return -f
#range = [0,10], d = 4, value = -10.5363