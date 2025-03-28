import gurobipy as gp
from gurobipy import GRB
import numpy as np

def create_relaxed_parameters():
    """
    Create parameters with intentionally relaxed constraints to ensure feasibility
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define smaller dimensions to make the problem more tractable
    I = list(range(5))   # 5 user areas
    J = list(range(3))   # 3 data centers
    K = list(range(2))   # 2 query types
    T = list(range(3))   # 3 time slots
    R = list(range(2))   # 2 resource types
    
    # Create random parameters with careful scaling
    params = {}
    
    # Add sets
    params['I'] = I
    params['J'] = J
    params['K'] = K
    params['T'] = T
    params['R'] = R
    
    # 1. Query demand - keep this relatively small
    lmbda = {(i, k, t): np.random.randint(5, 15) for i in I for k in K for t in T}
    params['lambda'] = lmbda
    
    # 2. Token counts - moderate values
    h = {k: np.random.randint(10, 30) for k in K}  # Input tokens
    f = {k: np.random.randint(20, 50) for k in K}  # Output tokens
    params['h'] = h
    params['f'] = f
    
    # 3. Energy parameters - carefully scaled
    tau_in = {k: 0.001 for k in K}    # Very small energy per input token
    tau_out = {k: 0.002 for k in K}   # Small energy per output token
    params['tau_in'] = tau_in
    params['tau_out'] = tau_out
    
    # 4. Cost parameters - uniform for simplicity
    gamma = {(j, t): 0.1 for j in J for t in T}    # Electricity price
    c = {(j, t): 0.1 for j in J for t in T}        # Local marginal price
    theta = {j: 0.2 for j in J}                    # Carbon intensity
    delta = {(j, t): 0.05 for j in J for t in T}   # Carbon tax
    
    params['gamma'] = gamma
    params['c'] = c
    params['theta'] = theta
    params['delta'] = delta
    
    # 5. Energy availability - ensure plenty of renewable energy
    P_w = {(j, t): np.random.uniform(200, 500) for j in J for t in T}  # Renewable energy
    P_max = {(j, t): 1000 for j in J for t in T}                       # Grid capacity (very high)
    params['P_w'] = P_w
    params['P_max'] = P_max
    
    # 6. Server parameters - ensure enough capacity
    p_idle = {j: 50 for j in J}                                     # Idle power
    p_peak = {j: 200 for j in J}                                    # Peak power
    m = {(j, t): np.random.randint(20, 50) for j in J for t in T}   # Number of servers
    H = {j: 50 for j in J}                                          # Queries per server (high)
    PUE = {j: 1.2 for j in J}                                       # Power Usage Effectiveness
    
    params['p_idle'] = p_idle
    params['p_peak'] = p_peak
    params['m'] = m
    params['H'] = H
    params['PUE'] = PUE
    
    # 7. Resource parameters - ensure enough capacity
    C = {(r, j): 1000 for r in R for j in J}                            # Resource capacity (very high)
    alpha = {(k, r): np.random.uniform(0.01, 0.05) for k in K for r in R}  # Resource requirement (low)
    params['C'] = C
    params['alpha'] = alpha
    
    # 8. Model placement - initially placed everywhere
    z_prev = {(j, k, t): 1 for j in J for k in K for t in T}
    f_download = {j: 10 for j in J}
    params['z_prev'] = z_prev
    params['f_download'] = f_download
    
    # 9. QoS parameters - relaxed constraints
    s = {(i, k, t): 5 for i in I for k in K for t in T}   # Unmet penalty (low)
    Gamma = {i: 0.5 for i in I}                           # Max unmet ratio (high = 50%)
    e = {k: 0.05 for k in K}                              # Error rate (low)
    E = {k: 0.7 for k in K}                               # Min accuracy (relaxed)
    
    params['s'] = s
    params['Gamma'] = Gamma
    params['e'] = e
    params['E'] = E
    
    # 10. Delay parameters - relaxed constraints
    beta = {(i, k, t): 0.5 for i in I for k in K for t in T}   # Token size
    B = {(i, j): 500 for i in I for j in J}                    # Bandwidth (high)
    d = {(i, j): 0.01 for i in I for j in J}                   # Network delay (low)
    v = {(j, t): 0.001 for j in J for t in T}                  # Processing delay (low)
    Delta = {(i, k, t): 10 for i in I for k in K for t in T}   # Delay threshold (very high)
    
    params['beta'] = beta
    params['B'] = B
    params['d'] = d
    params['v'] = v
    params['Delta'] = Delta
    
    # 11. Water parameters - relaxed constraints
    WUE = {(j, t): 0.2 for j in J for t in T}
    EWIF = {(j, t): 0.01 for j in J for t in T}
    Z = 10000  # Very high limit
    
    params['WUE'] = WUE
    params['EWIF'] = EWIF
    params['Z'] = Z
    
    return params

def test_feasibility(params):
    # Extract sets
    I = params['I']
    J = params['J']
    K = params['K']
    T = params['T']
    R = params['R']
    
    # Create a simplified model just to test feasibility
    model = gp.Model("Feasibility-Test")
    
    # Add key decision variables
    x = model.addVars([(i, j, k, t) for i in I for j in J for k in K for t in T], 
                      vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    z = model.addVars([(j, k, t) for j in J for k in K for t in T], 
                     vtype=GRB.BINARY, name="z")
    q = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                     vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    
    # Add critical constraints
    model.addConstrs((q[i, k, t] + sum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")
    model.addConstrs((x[i, j, k, t] <= z[j, k, t] for i in I for j in J for k in K for t in T),name="model_placement")
    model.addConstrs(
        (q[i, k, t] <= params['Gamma'][i] 
         for i in I for k in K for t in T),
        name="qos"
    )
    for j in J:
        for r in R:
            for t in T:
                model.addConstr(
                    gp.quicksum(
                        params['alpha'][k, r] * params['f'][k] * params['lambda'][i, k, t] * x[i, j, k, t] 
                        for i in I for k in K
                    ) <= params['C'][r, j],
                    f"resource_{j}_{r}_{t}"
                )
    
    # Add a simple objective (minimize unmet demand)
    model.setObjective(
        gp.quicksum(q[i, k, t] for i in I for k in K for t in T),
        GRB.MINIMIZE
    )
    
    # Set parameters for quick feasibility check
    model.setParam('OutputFlag', 0)  # Mute solver output
    model.setParam('TimeLimit', 60)
    
    # Try to find feasible solution
    model.optimize()
    
    # Check result
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            print("✅ FEASIBLE SOLUTION FOUND!")
            print(f"Objective value: {model.objVal}")
            return True, model
    else:
        print("❌ NO FEASIBLE SOLUTION FOUND")
        print(f"Status: {model.status}")
        
        # Compute IIS to find conflicting constraints
        if model.status == GRB.INFEASIBLE:
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            model.computeIIS()
            print("Conflicting constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  - {c.ConstrName}")
        
        return False, model

def simplified_green_llm(params):
    # Extract sets
    I = params['I']
    J = params['J']
    K = params['K']
    T = params['T']
    R = params['R']
    
    # Extract key parameters
    lmbda = params['lambda']
    h = params['h']
    f = params['f']
    s = params['s']
    Gamma = params['Gamma']
    alpha = params['alpha']
    C = params['C']
    
    # Create model
    model = gp.Model("Simplified-Green-LLM")
    
    # Decision variables
    x = model.addVars([(i, j, k, t) for i in I for j in J for k in K for t in T], 
                      vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    z = model.addVars([(j, k, t) for j in J for k in K for t in T], 
                     vtype=GRB.BINARY, name="z")
    q = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                     vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    
    # Workload allocation
    model.addConstrs(
        (q[i, k, t] + gp.quicksum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")
    
    # Model placement
    model.addConstrs(
        (x[i, j, k, t] <= z[j, k, t] 
         for i in I for j in J for k in K for t in T),
        name="model_placement"
    )
    
    # QoS limits
    model.addConstrs(
        (q[i, k, t] <= Gamma[i] 
         for i in I for k in K for t in T),
        name="qos"
    )
    
    # Resource limits
    model.addConstrs(
        (gp.quicksum(alpha[k, r] * f[k] * lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) <= C[r, j] for j in J for r in R for t in T),
        name="resource_capacity"
    )
    
    # Objective: minimize unmet demand cost + model placement cost
    obj = gp.quicksum(s[i, k, t] * lmbda[i, k, t] * q[i, k, t] for i in I for k in K for t in T)
    obj += gp.quicksum(z[j, k, t] for j in J for k in K for t in T) * 0.1  # Small cost for model placement
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Set solver parameters
    model.setParam('OutputFlag', 0)  # Mute solver output
    model.setParam('TimeLimit', 300)
    model.setParam('MIPGap', 0.1)
    
    # Optimize
    model.optimize()
    
    # Process results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
        if model.SolCount > 0:
            print(f"Solution found with objective: {model.objVal}")
            
            # Extract basic solution
            solution = {
                'x': {(i, j, k, t): x[i, j, k, t].X for i in I for j in J for k in K for t in T},
                'z': {(j, k, t): z[j, k, t].X for j in J for k in K for t in T},
                'q': {(i, k, t): q[i, k, t].X for i in I for k in K for t in T},
                'objective': model.objVal,
                'status': model.status
            }
            
            # Print decision variables
            print("Decision Variables:")
            print("x:", solution['x'])
            print("z:", solution['z'])
            print("q:", solution['q'])
            
            # Compute some statistics
            unmet = sum(q[i, k, t].X * lmbda[i, k, t] for i in I for k in K for t in T)
            total = sum(lmbda[i, k, t] for i in I for k in K for t in T)
            unmet_ratio = unmet / total if total > 0 else 0
            
            models_placed = sum(z[j, k, t].X for j in J for k in K for t in T)
            total_possible = len(J) * len(K) * len(T)
            
            print(f"Unmet demand: {unmet:.2f} / {total} queries ({unmet_ratio*100:.2f}%)")
            print(f"Models placed: {models_placed} / {total_possible} possible placements")
            
            return model, solution
    else:
        print(f"Failed to find solution. Status: {model.status}")
        return model, None

# Test the simplified model
if __name__ == "__main__":
    params = create_relaxed_parameters()
    feasible, model = test_feasibility(params)
    
    if feasible:
        # Try the simplified model
        model, solution = simplified_green_llm(params)