import gurobipy as gp
from gurobipy import GRB
import numpy as np

def green_llm_optimization(params):
    """
    Implement the Green-LLM optimization model using Gurobi
    
    Args:
        params: Dictionary containing all necessary parameters
        
    Returns:
        model: Solved Gurobi model
        solution: Dictionary containing optimal decision variables
    """
    # Extract parameters
    I = params['I']  # Set of user areas
    J = params['J']  # Set of data centers
    K = params['K']  # Set of query types
    T = params['T']  # Set of time slots
    R = params['R']  # Set of resource types
    
    # Query parameters
    lmbda = params['lambda']      # Number of type k queries from area i at time t
    h = params['h']               # Average input token count for type k queries
    f = params['f']               # Average output token count for type k queries
    
    # Energy parameters
    tau_in = params['tau_in']     # Energy consumption coefficient for input tokens
    tau_out = params['tau_out']   # Energy consumption coefficient for output tokens
    gamma = params['gamma']       # Electricity price at DC j at time t
    c = params['c']               # Local marginal price at DC j at time t
    theta = params['theta']       # Carbon intensity at DC j
    delta = params['delta']       # Carbon tax for DC j at time t
    
    # Data center parameters
    P_w = params['P_w']          # Renewable energy generated at DC j at time t
    P_max = params['P_max']      # Maximum power from grid for DC j at time t
    p_idle = params['p_idle']    # Idle power consumption of DC j
    p_peak = params['p_peak']    # Peak power consumption of DC j
    m = params['m']              # Number of active servers at DC j at time t
    H = params['H']              # Maximum queries per server at DC j
    PUE = params['PUE']          # Power Usage Effectiveness of DC j


    # Resource parameters
    C = params['C']              # Capacity of type r resource at DC j
    alpha = params['alpha']      # Type r resource required for processing type k tokens
    
    # Model placement parameters
    z_prev = params['z_prev']    # Known placement status from previous time slot
    f_download = params['f_download']  # Submodel download cost at DC j
    
    # QoS parameters
    s = params['s']              # Unmet penalty for type k query in area i at time t
    Gamma = params['Gamma']      # Maximum allowed unmet demand ratio for area i
    e = params['e']              # Error rate for processing type k queries
    E = params['E']              # Minimum required accuracy for type k queries
    
    # Delay parameters
    beta = params['beta']        # Average token size at time t
    B = params['B']              # Available bandwidth between i and j
    d = params['d']              # Network delay between area i and DC j
    v = params['v']              # Processing delay at DC j at time t
    Delta = params['Delta']      # Threshold on average round delay
    
    # Water parameters
    WUE = params['WUE']          # Water Usage Effectiveness of DC j at time t
    EWIF = params['EWIF']        # Energy-Water Intensity Factor of DC j at time t
    Z = params['Z']              # Maximum allowed water consumption
    
    # Create a new model
    model = gp.Model("Green-LLM")
    
    # Create decision variables using addVars for cleaner implementation
    x = model.addVars(I, J, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    z = model.addVars(J, K, T, vtype=GRB.BINARY, name="z")
    q = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    # Helper variables
    # eta = model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    # Helper variables
    P_d = model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    
    # Update model to integrate new variables
    model.update()
    
    # Objective function components
    
    # C1: Energy consumption cost
    C1 = gp.quicksum((tau_in[k] * h[k] * lmbda[i, k, t] * x[i, j, k, t] + tau_out[k] * f[k] * lmbda[i, k, t] * x[i, j, k, t]) * gamma[j, t] for i in I for j in J for k in K for t in T)
    
    # C2: Power procurement cost
    C2 = gp.quicksum(c[j, t] * P_g[j, t] for j in J for t in T)
    
    # C3: Carbon tax
    C3 = gp.quicksum(delta[j, t] * theta[j] * P_g[j, t] for j in J for t in T)
    
    # C4: Model placement and storage cost
    C4 = gp.quicksum(f_download[j] * (1 - z_prev[j, k, t]) * z[j, k, t] for j in J for k in K for t in T)
    
    # C5: Unmet demand penalty
    C5 = gp.quicksum(s[i, k, t] * lmbda[i, k, t] * q[i, k, t] for i in I for k in K for t in T)
    
    # Set the objective (minimize total cost)
    model.setObjective(C1 + C2 + C3 + C4 + C5, GRB.MINIMIZE)
    
    # Add constraints
    
    # for j in J:
    #     for t in T:
    #         eta = gp.quicksum(lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) / (m[j, t] * H[j])
    #         # Power consumption of DC (Equation 6a)
    #         model.addConstr(P_d[j, t] == m[j, t] * (p_idle[j] + eta * (p_peak[j] - p_idle[j])),f"power_consumption_{j}_{t}")
    

    # Server utilization calculation and power consumption (Equation 6) 
    # model.addConstrs((eta[j, t] == gp.quicksum(lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) / (m[j, t] * H[j]) for j in J for t in T), name="utilization")
    # Set the power consumption constraints
    eta = {(j, t): (lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) / (m[j, t] * H[j]) for j in J for t in T}
    model.addConstrs((P_d[j, t] == m[j, t] * (p_idle[j] + eta[j, t] * (p_peak[j] - p_idle[j])) for j in J for t in T), name="power_consumption")

    # Power balance constraints (Equation 7)
    model.addConstrs((P_d[j, t] == P_g[j, t] + P_w[j, t] for j in J for t in T),name="power_balance")
    
    # Grid capacity condition (Equation 8)
    model.addConstrs((P_g[j, t] <= P_max[j, t] for j in J for t in T), name="grid_capacity")
    
    # Long-term water footprint constraints (Equation 10)
    water_consumption = {(j, t): (WUE[j, t] / PUE[j] + EWIF[j, t]) * P_d[j, t] for j in J for t in T}
    model.addConstr(gp.quicksum(water_consumption[j, t] for j in J for t in T) <= Z,"long_term_water_footprint")
    
    # Resource capacity constraints (Equation 11)
    # Ensure workload allocation only if model is placed (Equation 11a)
    model.addConstrs((x[i, j, k, t] <= 1 - z[j, k, t] for i in I for j in J for k in K for t in T), name="model_placement")
    
    # Ensure resource capacity is not exceeded (Equation 11b)
    model.addConstrs((gp.quicksum(alpha[k, r] * f[k] * lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) <= C[r, j] for j in J for r in R for t in T), name="resource_capacity")
    
    # Workload allocation constraints (Equation 12)
    model.addConstrs((q[i, k, t] + gp.quicksum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")
    


    # Create variables for each type of delay
    D_tran = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, name="D_tran")
    D_prop = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, name="D_prop")
    D_proc = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, name="D_proc")

# Define transmission delay (Equation 13)
    model.addConstrs((D_tran[i, k, t] == gp.quicksum(beta[i, k, t] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) / B[i, j] for j in J) for i in I for k in K for t in T), name="trans_delay_def")

# Define propagation delay (Equation 14)
    model.addConstrs((D_prop[i, k, t] == gp.quicksum(d[i, j] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) for j in J) for i in I for k in K for t in T), name="prop_delay_def")

# Define processing delay (Equation 15)
    model.addConstrs((D_proc[i, k, t] == gp.quicksum(v[j, t] * f[k] * lmbda[i, k, t] * x[i, j, k, t] for j in J) for i in I for k in K for t in T),name="proc_delay_def")

# Total delay constraint (Equation 16)
    model.addConstrs((D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t] <= Delta[i, k, t] for i in I for k in K for t in T), name="max_delay")


    # for i in I:
    #     for k in K:
    #         for t in T:
    #             # Transmission delay (Equation 13)
    #             D_tran = gp.quicksum(
    #                 beta[i, k, t] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) / B[i, j]
    #                 for j in J
    #             )
                
    #             # Propagation delay (Equation 14)
    #             D_prop = gp.quicksum(
    #                 d[i, j] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k])
    #                 for j in J
    #             )
                
    #             # Processing delay (Equation 15)
    #             D_proc = gp.quicksum(
    #                 v[j, t] * f[k] * lmbda[i, k, t] * x[i, j, k, t]
    #                 for j in J
    #             )
                
    #             # Total delay constraint
    #             model.addConstr(
    #                 D_tran + D_prop + D_proc <= Delta[i, k, t],
    #                 f"max_delay_{i}_{k}_{t}"
    #             )
    
    # QoS constraints (Equation 17)
    model.addConstrs((q[i, k, t] <= Gamma[i] for i in I for k in K for t in T), name="qos")
    
    # Accuracy constraints (Equation 18)
    model.addConstrs((gp.quicksum((1 - e[k]) * lmbda[i, k, t] * x[i, j, k, t] for i in I for j in J) >= E[k] for k in K for t in T), name="accuracy")
    
    # Set optimization parameters
    model.setParam('OutputFlag', 1)  # Display optimization details
    model.setParam('TimeLimit', 3600)  # 1-hour time limit
    
    # Optimize the model
    model.optimize()
    
    # Check if optimal solution found
    if model.status == GRB.OPTIMAL:
        # Extract solution
        solution = {
            'x': {(i, j, k, t): x[i, j, k, t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j, t): P_g[j, t].X for j in J for t in T},
            'z': {(j, k, t): z[j, k, t].X for j in J for k in K for t in T},
            'q': {(i, k, t): q[i, k, t].X for i in I for k in K for t in T},
            'objective_value': model.objVal,
            'objective_components': {
                'C1': C1.getValue(),
                'C2': C2.getValue(),
                'C3': C3.getValue(),
                'C4': C4.getValue(),
                'C5': C5.getValue()
            }
        }
        return model, solution
    else:
        print(f"Optimization failed with status {model.status}")
        return model, None

def create_sample_params():
    """
    Create a sample parameter dictionary for testing the model
    """
    # Define small test sets
    I = list(range(5))  # 2 user areas
    J = list(range(2))  # 2 data centers
    K = list(range(2))  # 2 query types
    T = list(range(3))  # 3 time slots
    R = list(range(2))  # 2 resource types (e.g., CPU, GPU)
    
    # Create random parameters (in practice, use real data)
    np.random.seed(42)
    
    # Query parameters - using consistent ordering (i,j,k,t)
    lmbda = {(i, k, t): np.random.randint(50, 100) 
             for i in I for k in K for t in T}
    h = {k: np.random.randint(10, 50) for k in K}
    f = {k: np.random.randint(50, 200) for k in K}
    # Length of token
    tau_in = {k: np.random.uniform(0.001, 0.01) for k in K}
    tau_out = {k: np.random.uniform(0.001, 0.01) for k in K}  
    # Energy parameters

    gamma = {(j, t): np.random.uniform(0.05, 0.2) for j in J for t in T}
    c = {(j, t): np.random.uniform(5, 15) for j in J for t in T}
    theta = {j: np.random.uniform(0.2, 0.6) for j in J}
    delta = {(j, t): np.random.uniform(20, 50) for j in J for t in T}
    
    # Data center parameters
    P_w = {(j, t): np.random.uniform(100, 500) for j in J for t in T}
    P_max = {(j, t): np.random.uniform(1000, 2000) for j in J for t in T}
    p_idle = {j: np.random.uniform(50, 100) for j in J}
    p_peak = {j: np.random.uniform(300, 500) for j in J}
    m = {(j, t): np.random.randint(10, 50) for j in J for t in T}
    H = {j: np.random.randint(5, 15) for j in J}
    PUE = {j: np.random.uniform(1.1, 1.5) for j in J}
    
    # Resource parameters
    C = {(r, j): np.random.randint(500, 1000) for r in R for j in J}
    alpha = {(k, r): np.random.uniform(0.01, 0.1) for k in K for r in R}
    
    # Model placement parameters
    z_prev = {(j, k, t): np.random.randint(0, 2) for j in J for k in K for t in T}
    f_download = {j: np.random.uniform(10, 30) for j in J}
    
    # QoS parameters
    s = {(i, k, t): np.random.uniform(5, 20) for i in I for k in K for t in T}
    Gamma = {i: np.random.uniform(0.05, 0.2) for i in I}
    e = {k: np.random.uniform(0.01, 0.1) for k in K}
    E = {k: np.random.uniform(0.8, 0.95) for k in K}
    
    # Delay parameters
    beta = {(i, k, t): np.random.uniform(1, 5) for i in I for k in K for t in T}
    B = {(i, j): np.random.uniform(50, 200) for i in I for j in J}
    d = {(i, j): np.random.uniform(0.01, 0.1) for i in I for j in J}
    v = {(j, t): np.random.uniform(0.001, 0.01) for j in J for t in T}
    Delta = {(i, k, t): np.random.uniform(0.5, 2.0) for i in I for k in K for t in T}
    
    # Water parameters
    WUE = {(j, t): np.random.uniform(0.5, 2.0) for j in J for t in T}
    EWIF = {(j, t): np.random.uniform(0.01, 0.05) for j in J for t in T}
    Z = np.random.uniform(5000, 10000)
    
    params = {
        'I': I, 'J': J, 'K': K, 'T': T, 'R': R,
        'lambda': lmbda, 'h': h, 'f': f,
        'tau_in': tau_in, 'tau_out': tau_out, 
        'gamma': gamma, 'c': c,
        'theta': theta, 'delta': delta,
        'P_w': P_w, 'P_max': P_max,
        'p_idle': p_idle, 'p_peak': p_peak,
        'm': m, 'H': H, 'PUE': PUE,
        'C': C, 'alpha': alpha,
        'z_prev': z_prev, 'f_download': f_download,
        's': s, 'Gamma': Gamma,
        'e': e, 'E': E,
        'beta': beta, 'B': B,
        'd': d, 'v': v,
        'Delta': Delta,
        'WUE': WUE, 'EWIF': EWIF, 'Z': Z
    }
    
    return params

def analyze_results(solution):
    """
    Analyze and display the optimization results
    """
    if solution is None:
        print("No solution to analyze")
        return
    
    print("\n===== Green-LLM Optimization Results =====")
    print(f"Total Cost: {solution['objective_value']:.2f}")
    print("\nCost Components:")
    for component, value in solution['objective_components'].items():
        print(f"  {component}: {value:.2f}")
    
    # Additional analysis could include:
    # - Average workload allocation by DC
    # - Renewable energy utilization
    # - Carbon emission reduction
    # - Water savings
    # etc.

# Example usage
if __name__ == "__main__":
    try:
        # Create sample parameters
        params = create_sample_params()
        
        # Solve the model
        model, solution = green_llm_optimization(params)
        
        # Analyze results
        if solution:
            analyze_results(solution)
            
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
    except Exception as e:
        print(f"Error: {e}")