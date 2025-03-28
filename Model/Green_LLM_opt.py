import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd



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
    x = model.addVars([(i, j, k, t) for i in I for j in J for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    z = model.addVars([(j, k, t) for j in J for k in K for t in T], vtype=GRB.BINARY, name="z")
    q = model.addVars([(i, k, t) for i in I for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    
    # Create variables for each type of delay
    D_tran = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                          vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                          vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                          vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    
    # Water consumption variable
    woc = model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="woc")
    
    # Helper variables
    eta = model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="eta")
    P_d = model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    
    # Update model to integrate new variables
    model.update()
    
    # Objective function components
    
    # C1: Energy consumption cost
    C1 = gp.quicksum((tau_in[k] * h[k] * lmbda[i, k, t] * x[i, j, k, t] + tau_out[k] * f[k] * lmbda[i, k, t] * x[i, j, k, t]) * gamma[j, t] for i in I for j in J for k in K for t in T)
    
    # C2: Power procurement cost
    C2 = gp.quicksum(c[j, t] * P_g[j, t] for j in J for t in T)
    
    # C3: Carbon tax
    C3 = gp.quicksum(delta[j, t] * theta[j] * P_g[j, t] for j in J for t in T)
    
    # C4: Model placement and storage cost (avoiding redundant downloads)
    C4_terms = []
    for j in J:
        for k in K:
            for t in T:
                if t == 0:
                    # For the first time slot, use the provided z_prev
                    C4_terms.append(f_download[j] * (1 - z_prev[j, k, t]) * z[j, k, t])
                else:
                    # For subsequent time slots, use the decision from previous time
                    C4_terms.append(f_download[j] * (1 - z[j, k, t-1]) * z[j, k, t])
    
    C4 = gp.quicksum(C4_terms)
    
    # C5: Unmet demand penalty
    C5 = gp.quicksum(s[i, k, t] * lmbda[i, k, t] * q[i, k, t] for i in I for k in K for t in T)
    
    # Set the objective (minimize total cost)
    model.setObjective(C1 + C2 + C3 + C4 + C5, GRB.MINIMIZE)
    
    # Add constraints
    
    # Server utilization calculation (Equation 6) 
    for j in J:
        for t in T:
            # Prevent division by zero for server utilization
            if m[j, t] * H[j] > 0:
                model.addConstr(
                    eta[j, t] == gp.quicksum(lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) / (m[j, t] * H[j]),
                    f"utilization_{j}_{t}"
                )
            else:
                # If no servers available, force utilization to zero
                model.addConstr(eta[j, t] == 0, f"utilization_zero_{j}_{t}")
                # Also force allocation to this DC to be zero
                for i in I:
                    for k in K:
                        model.addConstr(x[i, j, k, t] == 0, f"no_allocation_{i}_{j}_{k}_{t}")
    
    # Power consumption calculation (Equation 6a)
    model.addConstrs((P_d[j, t] == m[j, t] * (p_idle[j] + eta[j, t] * (p_peak[j] - p_idle[j])) for j in J for t in T),name="power_consumption")

    # Power balance constraints (Equation 7)
    model.addConstrs((P_d[j, t] <= P_g[j, t] + P_w[j, t] for j in J for t in T), name="power_balance")
    
    # Grid capacity condition (Equation 8)
    model.addConstrs((P_g[j, t] <= P_max[j, t] for j in J for t in T), name="grid_capacity")
    
    # Long-term water footprint constraints (Equation 10)
    model.addConstrs((woc[j, t] == (WUE[j, t] / PUE[j] + EWIF[j, t]) * P_d[j, t] for j in J for t in T),name="water_consumption")
    
    model.addConstr(gp.quicksum(woc[j, t] for j in J for t in T) <= 1000 * Z, name="long_term_water_footprint")
    
    # Resource capacity constraints (Equation 11)
    
    # Ensure workload allocation only if model is placed (Equation 11a)
    # Note: This constraint is actually x[i,j,k,t] <= z[j,k,t] not 1-z[j,k,t]
    # since z[j,k,t]=1 means the model IS placed at DC j
    model.addConstrs((x[i, j, k, t] <= z[j, k, t] for i in I for j in J for k in K for t in T),name="model_placement")
    
    # Ensure resource capacity is not exceeded (Equation 11b)
    model.addConstrs((gp.quicksum(alpha[k, r] * f[k] * lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) <=  C[r, j] for j in J for r in R for t in T),name="resource_capacity")
    
    # Workload allocation constraints (Equation 12)
    model.addConstrs(
        (q[i, k, t] + gp.quicksum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T),
        name="workload_allocation"
    )
    
    # Define transmission delay (Equation 13)
    for i in I:
        for k in K:
            for t in T:
                # Add a small epsilon to avoid division by zero in bandwidth
                model.addConstr(
                    D_tran[i, k, t] == gp.quicksum(
                        beta[i, k, t] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) / max(0.001, B[i, j])
                        for j in J
                    ),
                    f"trans_delay_def_{i}_{k}_{t}"
                )

    # Define propagation delay (Equation 14)
    model.addConstrs(
        (D_prop[i, k, t] == gp.quicksum(
            d[i, j] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) for j in J)
         for i in I for k in K for t in T),
        name="prop_delay_def"
    )

    # Define processing delay (Equation 15)
    model.addConstrs(
        (D_proc[i, k, t] == gp.quicksum(
            v[j, t] * f[k] * lmbda[i, k, t] * x[i, j, k, t] for j in J)
         for i in I for k in K for t in T),
        name="proc_delay_def"
    )

    # Total delay constraint (Equation 16)
    model.addConstrs((D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t] <= Delta[i, k, t] for i in I for k in K for t in T), name="max_delay")
    
   #  QoS constraints (Equation 17)
    model.addConstrs((q[i, k, t] <= Gamma[i] for i in I for k in K for t in T), name="qos")
    
    # Accuracy constraints (Equation 18)
    # Modified to handle potential infeasibility - make sure accuracy requirement is proportional to satisfied demand
    for k in K:
        for t in T:
            total_demand = gp.quicksum(lmbda[i, k, t] for i in I)
            satisfied_demand = gp.quicksum((1 - q[i, k, t]) * lmbda[i, k, t] for i in I)
            model.addConstr(gp.quicksum((1 - e[k]) * lmbda[i, k, t] * x[i, j, k, t] for i in I for j in J) >= E[k] * satisfied_demand,
                f"accuracy_{k}_{t}"
            )
    
    # Set optimization parameters
    model.setParam('OutputFlag', 0)        # Display optimization details
    model.setParam('TimeLimit', 600)       # 10-minute time limit
    model.setParam('MIPGap', 0.05)         # Accept solutions within 5% of optimal
    model.setParam('FeasibilityTol', 1e-5) # Slightly relaxed feasibility tolerance
    model.setParam('NumericFocus', 3)      # Maximum numeric stability
    
    # Optional: Add warm start if available from previous solution
    if 'previous_solution' in params:
        prev_sol = params['previous_solution']
        for i in I:
            for j in J:
                for k in K:
                    for t in T:
                        if (i, j, k, t) in prev_sol['x']:
                            x[i, j, k, t].start = prev_sol['x'][i, j, k, t]
    
    # Optimize the model
    try:
        model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return model, None
    
    # Check if optimal solution found
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
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
            },
            'status': model.status
        }
        
        # Add solution quality metrics
        if model.status == GRB.OPTIMAL:
            solution['quality'] = "Optimal"
            print(f"Optimization status: {solution['quality']}")
            print(f"Objective value: {solution['objective_value']:.2f}")
            print(f"Energy cost: {solution['objective_components']['C1']:.2f}")
        elif model.status == GRB.TIME_LIMIT:
            solution['quality'] = f"Time limit reached. Gap: {model.MIPGap*100:.2f}%"
        elif model.status == GRB.SUBOPTIMAL:
            solution['quality'] = "Suboptimal solution found"
            
        return model, solution
    else:
        status_codes = {
            GRB.LOADED: "Model is loaded, but not solved",
            GRB.INFEASIBLE: "Model is infeasible",
            GRB.INF_OR_UNBD: "Model is infeasible or unbounded",
            GRB.UNBOUNDED: "Model is unbounded",
            GRB.CUTOFF: "Objective values worse than cutoff",
            GRB.ITERATION_LIMIT: "Iteration limit reached",
            GRB.NODE_LIMIT: "Node limit reached",
            GRB.SOLUTION_LIMIT: "Solution limit reached",
            GRB.INTERRUPTED: "Optimization was interrupted",
            GRB.NUMERIC: "Numerical difficulties encountered",
            GRB.INPROGRESS: "Optimization in progress",
            GRB.USER_OBJ_LIMIT: "User objective limit reached"
        }
        
        status_msg = status_codes.get(model.status, f"Unknown status: {model.status}")
        print(f"Optimization failed: {status_msg}")
        
        # If infeasible, try to find the source of infeasibility
        if model.status == GRB.INFEASIBLE:
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
            
        return model, None
    
# Run the optimization
# model, solution = green_llm_optimization(params)

