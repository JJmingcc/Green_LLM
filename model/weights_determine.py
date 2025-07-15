import gurobipy as gp
from gurobipy import GRB
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def green_llm_model(params,sigma_e,sigma_c,sigma_d):
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
    rho = params['rho']           # Delay penalty parameter for type k queries
    # Energy parameters
    tau_in = params['tau_in']     # Energy consumption coefficient for input tokens
    tau_out = params['tau_out']   # Energy consumption coefficient for output tokens
    c = params['c']               # Local marginal price at DC j at time t
    theta = params['theta']       # Carbon intensity at DC j
    delta = params['delta']       # Carbon tax for DC j at time t
    
    # Data center parameters
    P_w = params['P_w']          # Renewable energy generated at DC j at time t
    P_max = params['P_max']      # Maximum power from grid for DC j at time t
    PUE = params['PUE']          # Power Usage Effectiveness of DC j

    # Resource parameters
    C = params['C']              # Capacity of type r resource at DC j
    alpha = params['alpha']      # Type r resource required for processing type k tokens

    
    
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
    m0_model = gp.Model("Green-LLM")
    
    # Create decision variables using addVars for cleaner implementation
    x = m0_model.addVars([(i, j, k, t) for i in I for j in J for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = m0_model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    # Create variables for each type of delay
    D_tran = m0_model.addVars([(i, k, t) for i in I for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = m0_model.addVars([(i, k, t) for i in I for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = m0_model.addVars([(i, k, t) for i in I for k in K for t in T], vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    # Water consumption variable
    woc = m0_model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="woc")
    # Helper variables
    eta = m0_model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="eta")
    P_d = m0_model.addVars([(j, t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    P_c = m0_model.addVars([(j,t) for j in J for t in T], vtype=GRB.CONTINUOUS, lb=0, name="P_c")
    # Update model to integrate new variables
    m0_model.update()
    
    # Objective function components
    
    #  Power procurement cost
    C_energy = gp.quicksum(c[j, t] * P_g[j, t] for j in J for t in T)

    # Carbon tax
    C_carbon = gp.quicksum(delta[j] * theta[j,t] * P_g[j, t] for j in J for t in T)
   
    C_delay = gp.quicksum(rho[k] * (D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t]) for i in I for k in K for t in T)

    
    # Set the objective (minimize total cost) with weighted components
    m0_model.setObjective(sigma_e*C_energy + sigma_c*C_carbon + sigma_d*C_delay, GRB.MINIMIZE)

    # Computational energy consumption
    m0_model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) for j in J for t in T), name="computation_consumption")
    
    # Total Energy consumption (including cooling and other overheads)
    m0_model.addConstrs((P_d[j, t] == PUE[j] * P_c[j, t] for j in J for t in T), name="tot_power_consumption")

    # Power balance constraints (Equation 7)
    m0_model.addConstrs((P_d[j, t] <= P_g[j, t] + P_w[j, t] for j in J for t in T), name="power_balance")
    
    # Grid capacity condition (Equation 8)
    m0_model.addConstrs((P_g[j, t] <= P_max[j, t] for j in J for t in T), name="grid_capacity")
    
    # Long-term water footprint constraints (Equation 10)
    m0_model.addConstrs((woc[j, t] == (WUE[j, t] / PUE[j] + EWIF[j, t]) * P_d[j, t] for j in J for t in T),name="water_consumption")
    m0_model.addConstr(gp.quicksum(woc[j, t] for j in J for t in T) <= Z, name="long_term_water_footprint")
    
    
    
    # Ensure resource capacity is not exceeded (Equation 11b)
    m0_model.addConstrs((gp.quicksum(alpha[k, r] * (f[k]+h[k]) * lmbda[i, k, t] * x[i, j, k, t] for i in I for k in K) <=  C[r, j] for j in J for r in R for t in T),name="resource_capacity")
    
    # Workload allocation constraints (Equation 12)
    m0_model.addConstrs((gp.quicksum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")
    
    # Define transmission delay (Equation 13)
    for i in I:
        for k in K:
            for t in T:
                # Add a small epsilon to avoid division by zero in bandwidth
                m0_model.addConstr(
                    D_tran[i, k, t] == gp.quicksum(beta[i, k, t] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) / max(0.001, B[i, j]) for j in J),
                    f"trans_delay_def"
                )

    # Define propagation delay (Equation 14)
    m0_model.addConstrs((D_prop[i, k, t] == gp.quicksum(d[i, j] * x[i, j, k, t]  for j in J) for i in I for k in K for t in T), name="prop_delay_def")

    # Define processing delay (Equation 15)
    m0_model.addConstrs((D_proc[i, k, t] == gp.quicksum(v[j, k] * (f[k]+h[k])* lmbda[i,k,t] * x[i, j, k, t] for j in J) for i in I for k in K for t in T), name="proc_delay_def")

    # Total delay constraint (Equation 16)
    m0_model.addConstrs((D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t] <= Delta[i, k] for i in I for k in K for t in T), name="max_delay")
    

    
    # Set optimization parameters
    m0_model.setParam('OutputFlag', 0)        # Display optimization details
    m0_model.setParam('TimeLimit', 600)       # 10-minute time limit
    m0_model.setParam('MIPGap', 0.05)         # Accept solutions within 5% of optimal
    m0_model.setParam('FeasibilityTol', 1e-5) # Slightly relaxed feasibility tolerance
    m0_model.setParam('NumericFocus', 3)      # Maximum numeric stability
    
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
        m0_model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return m0_model, None
    
    # Check if optimal solution found
    if m0_model.status == GRB.OPTIMAL or m0_model.status == GRB.TIME_LIMIT or m0_model.status == GRB.SUBOPTIMAL:
        # Extract solution
        C_energy_list, C_carbon_list, woc_list, C_delay_list = [],[],[],[]
        # Cost components with time
        for t in T: 
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i,k,t].X + D_prop[i,k,t].X + D_proc[i,k,t].X) for i in I for k in K)) 
            woc_list.append(sum(woc[j,t].X for j in J))

        C_tot_with_time = C_energy_list + C_carbon_list + C_delay_list      
        C_tot = sum(C_energy_list) + sum(C_carbon_list) + sum(C_delay_list)


        m0_sol = {
            'x': {(i, j, k, t): x[i, j, k, t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j, t): P_g[j, t].X for j in J for t in T},
            'P_c': {(j, t): P_c[j, t].X for j in J for t in T},
            'P_d': {(j, t): P_d[j, t].X for j in J for t in T},
            'D_tran': {(i, k, t): D_tran[i, k, t].X for i in I for k in K for t in T},
            'D_prop': {(i, k, t): D_prop[i, k, t].X for i in I for k in K for t in T},
            'D_proc': {(i, k, t): D_proc[i, k, t].X for i in I for k in K for t in T},
            'woc': {(j, t): woc[j, t].X for j in J for t in T},
            'woc_with_time': woc_list,
            'C_energy_with_time': C_energy_list,
            'C_carbon_with_time': C_carbon_list,
            'C_delay_with_time': C_delay_list,
            'm0_delay': sum(C_delay_list),             
            'm0_cost_time': C_tot_with_time,
            'm0_cost': C_tot,
            'm0_obj': m0_model.objVal,
            'objective_components': {
                'C_energy': C_energy.getValue(),
                'C_carbon': C_carbon.getValue(),
                'C_delay': C_delay.getValue()
            },
            'status': m0_model.status
        }
        return m0_model, m0_sol
    else:
        print(f"Optimization failed: Status {m0_model.status}")
        if m0_model.status == GRB.INFEASIBLE:
            print("Computing IIS...")
            m0_model.computeIIS()
            m0_model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        return m0_model, None





