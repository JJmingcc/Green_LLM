import gurobipy as gp
from gurobipy import GRB
import numpy as np



def m1_model(params):
    """
    Implement the baseline model, minimizing the C1_cost only, energy-saving model

    Args:
        params: Dictionary containing all necessary parameters

    Returns:
        model: Solved Gurobi model
        solution: Dictionary containing optimal decision variables
    """
    # Extract parameters
    I, J, K, T, R = params['I'], params['J'], params['K'], params['T'], params['R']
    lmbda, h, f = params['lambda'], params['h'], params['f']
    tau_in, tau_out, c, theta, delta = params['tau_in'], params['tau_out'], params['c'], params['theta'], params['delta']
    P_w, P_max, PUE = params['P_w'], params['P_max'], params['PUE']
    C, alpha = params['C'], params['alpha']
    beta, B, d, v, Delta = params['beta'], params['B'], params['d'], params['v'], params['Delta']
    WUE, EWIF, Z = params['WUE'], params['EWIF'], params['Z']
    rho = params['rho']

    m1_model = gp.Model("Energy_saving_model")

    # Variables
    x = m1_model.addVars(I, J, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = m1_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    D_tran = m1_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = m1_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = m1_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    woc = m1_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="woc")
    eta = m1_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    P_d = m1_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    P_c = m1_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_c")

    m1_model.update()

    # Objective: minimize total delay
    C_energy = gp.quicksum(c[j, t] * P_g[j, t] for j in J for t in T)
    m1_model.setObjective(C_energy, GRB.MINIMIZE)

    # Constraints
    m1_model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) for j in J for t in T), name="computation_consumption")
    m1_model.addConstrs((P_d[j,t] == PUE[j] * P_c[j,t] for j in J for t in T), name="tot_power_consumption")
    m1_model.addConstrs((P_d[j,t] <= P_g[j,t] + P_w[j,t] for j in J for t in T), name="power_balance")
    m1_model.addConstrs((P_g[j,t] <= P_max[j,t] for j in J for t in T), name="grid_capacity")
    m1_model.addConstrs((woc[j,t] == (WUE[j,t] / PUE[j] + EWIF[j,t]) * P_d[j,t] for j in J for t in T), name="water_consumption")
    m1_model.addConstr(gp.quicksum(woc[j,t] for j in J for t in T) <= Z, name="long_term_water_footprint")
    m1_model.addConstrs((gp.quicksum(alpha[k,r] * (f[k] + h[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) <= C[r,j] for j in J for r in R for t in T), name="resource_capacity")
    m1_model.addConstrs((gp.quicksum(x[i,j,k,t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")
    for i in I:
        for k in K:
            for t in T:
                m1_model.addConstr(
                    D_tran[i,k,t] == gp.quicksum(beta[i,k,t] * lmbda[i,k,t] * x[i,j,k,t] * (h[k]+f[k]) / max(0.001,B[i,j]) for j in J),
                    name=f"trans_delay_def_{i}_{k}_{t}"
                )

    m1_model.addConstrs((D_prop[i,k,t] == gp.quicksum(d[i,j] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="prop_delay_def")
    m1_model.addConstrs((D_proc[i,k,t] == gp.quicksum(v[j,k] * (f[k]+h[k]) * lmbda[i,k,t] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="proc_delay_def")
    m1_model.addConstrs((D_tran[i,k,t] + D_prop[i,k,t] + D_proc[i,k,t] <= Delta[i,k] for i in I for k in K for t in T), name="max_delay")

    # Gurobi solver parameters
    m1_model.setParam('OutputFlag', 0)
    m1_model.setParam('TimeLimit', 600)
    m1_model.setParam('MIPGap', 0.05)
    m1_model.setParam('FeasibilityTol', 1e-5)
    m1_model.setParam('NumericFocus', 3)

    # Optimize
    try:
        m1_model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return m1_model, None

    if m1_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        # Extract solution
        C_energy_list, C_carbon_list, woc_list, C_delay_list, C_tot_with_time = [], [], [], [], []
        for t in T:
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i, k, t].X + D_prop[i, k, t].X + D_proc[i, k, t].X) for i in I for k in K))
            woc_list.append(sum(woc[j,t].X for j in J)) 
        
        C_tot_with_time = C_energy_list + C_carbon_list + C_delay_list      
        C_tot = sum(C_energy_list) + sum(C_carbon_list) + sum(C_delay_list)

        m1_sol = {
            'x': {(i,j,k,t): x[i,j,k,t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j,t): P_g[j,t].X for j in J for t in T},
            'P_c': {(j,t): P_c[j,t].X for j in J for t in T},
            'P_d': {(j,t): P_d[j,t].X for j in J for t in T},
            'D_tran': {(i,k,t): D_tran[i,k,t].X for i in I for k in K for t in T},
            'D_prop': {(i,k,t): D_prop[i,k,t].X for i in I for k in K for t in T},
            'D_proc': {(i,k,t): D_proc[i,k,t].X for i in I for k in K for t in T},
            'woc': {(j,t): woc[j,t].X for j in J for t in T},
            'woc_with_time': woc_list,
            'C_energy_with_time': C_energy_list,
            'C_carbon_with_time': C_carbon_list,
            'C_delay_with_time': C_delay_list,
            'm1_C_energy': sum(C_energy_list),
            'm1_C_carbon': sum(C_carbon_list),
            'm1_C_delay': sum(C_delay_list),
            'm1_cost_time': C_tot_with_time,
            'm1_cost': C_tot,
            'm1_obj': m1_model.objVal,
            'objective_components': {
                'C_energy': C_energy.getValue()
            },
            'status': m1_model.status
        }
        return m1_model, m1_sol
    else:
        print(f"Optimization failed: Status {m1_model.status}")
        if m1_model.status == GRB.INFEASIBLE:
            print("Computing IIS...")
            m1_model.computeIIS()
            m1_model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        return m1_model, None





def m2_model(params):
    """
    Implement the baseline model 3, minimizing the carbon-saving model

    Args:
        params: Dictionary containing all necessary parameters

    Returns:
        model: Solved Gurobi model
        solution: Dictionary containing optimal decision variables
    """     
    # Extract parameters
    I, J, K, T, R = params['I'], params['J'], params['K'], params['T'], params['R']
    lmbda, h, f = params['lambda'], params['h'], params['f']
    tau_in, tau_out, c, theta, delta = params['tau_in'], params['tau_out'], params['c'], params['theta'], params['delta']
    P_w, P_max, PUE = params['P_w'], params['P_max'], params['PUE']
    C, alpha = params['C'], params['alpha']
    beta, B, d, v, Delta = params['beta'], params['B'], params['d'], params['v'], params['Delta']
    WUE, EWIF, Z = params['WUE'], params['EWIF'], params['Z']
    rho = params['rho'] # delay penalty

    m2_model = gp.Model("Carbon_saving_model")

    # Variables
    x = m2_model.addVars(I, J, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = m2_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    D_tran = m2_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = m2_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = m2_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    woc = m2_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="woc")
    eta = m2_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    P_d = m2_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    P_c = m2_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_c")

    m2_model.update()

    # Objective: minimize the carbon intesity

    C_carbon = gp.quicksum(delta[j] * theta[j,t] * P_g[j, t] for j in J for t in T)
    m2_model.setObjective(C_carbon, GRB.MINIMIZE)

    # Constraints
    m2_model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) for j in J for t in T), name="computation_consumption")
    m2_model.addConstrs((P_d[j,t] == PUE[j] * P_c[j,t] for j in J for t in T), name="tot_power_consumption")
    m2_model.addConstrs((P_d[j,t] <= P_g[j,t] + P_w[j,t] for j in J for t in T), name="power_balance")
    m2_model.addConstrs((P_g[j,t] <= P_max[j,t] for j in J for t in T), name="grid_capacity")
    m2_model.addConstrs((woc[j,t] == (WUE[j,t] / PUE[j] + EWIF[j,t]) * P_d[j,t] for j in J for t in T), name="water_consumption")
    m2_model.addConstr(gp.quicksum(woc[j,t] for j in J for t in T) <= Z, name="long_term_water_footprint")
    m2_model.addConstrs((gp.quicksum(alpha[k,r] * (f[k] + h[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) <= C[r,j] for j in J for r in R for t in T), name="resource_capacity")
    m2_model.addConstrs((gp.quicksum(x[i,j,k,t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")

    for i in I:
        for k in K:
            for t in T:
                m2_model.addConstr(
                    D_tran[i,k,t] == gp.quicksum(beta[i,k,t] * lmbda[i,k,t] * x[i,j,k,t] * (h[k]+f[k]) / max(0.001,B[i,j]) for j in J),
                    name=f"trans_delay_def_{i}_{k}_{t}"
                )

    m2_model.addConstrs((D_prop[i,k,t] == gp.quicksum(d[i,j] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="prop_delay_def")
    m2_model.addConstrs((D_proc[i,k,t] == gp.quicksum(v[j,k] * (f[k]+h[k]) * lmbda[i,k,t] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="proc_delay_def")
    m2_model.addConstrs((D_tran[i,k,t] + D_prop[i,k,t] + D_proc[i,k,t] <= Delta[i,k] for i in I for k in K for t in T), name="max_delay")

    # Gurobi solver parameters
    m2_model.setParam('OutputFlag', 0)
    m2_model.setParam('TimeLimit', 600)
    m2_model.setParam('MIPGap', 0.05)
    m2_model.setParam('FeasibilityTol', 1e-5)
    m2_model.setParam('NumericFocus', 3)

    # Optimize
    try:
        m2_model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return m2_model, None

    if m2_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        # Extract solution
        C_energy_list, C_carbon_list, woc_list, C_delay_list, C_tot_with_time = [], [], [], [], []
        for t in T:
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i, k, t].X + D_prop[i, k, t].X + D_proc[i, k, t].X) for i in I for k in K))
            woc_list.append(sum(woc[j,t].X for j in J))
        
        C_tot_with_time = C_energy_list + C_carbon_list + C_delay_list      
        C_tot = sum(C_energy_list) + sum(C_carbon_list) + sum(C_delay_list)   

        m2_sol = {
            'x': {(i,j,k,t): x[i,j,k,t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j,t): P_g[j,t].X for j in J for t in T},
            'P_c': {(j,t): P_c[j,t].X for j in J for t in T},
            'P_d': {(j,t): P_d[j,t].X for j in J for t in T},
            'D_tran': {(i,k,t): D_tran[i,k,t].X for i in I for k in K for t in T},
            'D_prop': {(i,k,t): D_prop[i,k,t].X for i in I for k in K for t in T},
            'D_proc': {(i,k,t): D_proc[i,k,t].X for i in I for k in K for t in T},
            'woc': {(j,t): woc[j,t].X for j in J for t in T},
            'woc_with_time': woc_list,
            'C_energy_with_time': C_energy_list,
            'C_carbon_with_time': C_carbon_list,
            'C_delay_with_time': C_delay_list,
            'm2_C_energy': sum(C_energy_list),
            'm2_C_carbon': sum(C_carbon_list),
            'm2_C_delay': sum(C_delay_list),
            'm1_cost_time': C_tot_with_time,
            'm2_cost': C_tot,
            'm2_obj': m2_model.objVal,
            'objective_components': {
                'C_carbon': C_carbon.getValue()
            },
            'status': m2_model.status
        }
        return m2_model, m2_sol
    else:
        print(f"Optimization failed: Status {m2_model.status}")
        if m2_model.status == GRB.INFEASIBLE:
            print("Computing IIS...")
            m2_model.computeIIS()
            m2_model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        return m2_model, None
    


def m3_model(params):
    """
    Implement the baseline model 4, minimizing the number of DC used for LLM submodel placement
    Args:
        params: Dictionary containing all necessary parameters

    Returns:
        model: Solved Gurobi model
        solution: Dictionary containing optimal decision variables
    """
    # Extract parameters
    I, J, K, T, R = params['I'], params['J'], params['K'], params['T'], params['R']
    lmbda, h, f = params['lambda'], params['h'], params['f']
    tau_in, tau_out, c, theta, delta = params['tau_in'], params['tau_out'], params['c'], params['theta'], params['delta']
    P_w, P_max, PUE = params['P_w'], params['P_max'], params['PUE']
    C, alpha = params['C'], params['alpha']
    s_woc = params['s_woc']
    beta, B, d, v, Delta = params['beta'], params['B'], params['d'], params['v'], params['Delta']
    WUE, EWIF, Z = params['WUE'], params['EWIF'], params['Z']
    rho = params['rho']

    m3_model = gp.Model("hybrid_model")

    # Variables
    x = m3_model.addVars(I, J, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = m3_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    q = m3_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    D_tran = m3_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = m3_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = m3_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    woc = m3_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="woc")
    P_d = m3_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    P_c = m3_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_c")
    m3_model.update()

    C_delay = gp.quicksum(rho[k] * (D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t]) for i in I for k in K for t in T)
    C_energy = gp.quicksum(c[j,t] * P_g[j,t] for j in J for t in T)
    C_carbon = gp.quicksum(delta[j] * theta[j,t] * P_g[j,t] for j in J for t in T)
    m3_obj = 1/4*C_energy + 1/2*C_carbon + 1/4*C_delay
    m3_model.setObjective(m3_obj, GRB.MINIMIZE)
   

    # Constraints
    m3_model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) for j in J for t in T), name="computation_consumption")
    m3_model.addConstrs((P_d[j,t] == PUE[j] * P_c[j,t] for j in J for t in T), name="tot_power_consumption")
    m3_model.addConstrs((P_d[j,t] <= P_g[j,t] + P_w[j,t] for j in J for t in T), name="power_balance")
    m3_model.addConstrs((P_g[j,t] <= P_max[j,t] for j in J for t in T), name="grid_capacity")
    m3_model.addConstrs((woc[j,t] == (WUE[j,t] / PUE[j] + EWIF[j,t]) * P_d[j,t] for j in J for t in T), name="water_consumption")
    m3_model.addConstr(gp.quicksum(woc[j,t] for j in J for t in T) <= Z, name="long_term_water_footprint")
    m3_model.addConstrs((gp.quicksum(alpha[k,r] * (f[k] + h[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) <= C[r,j] for j in J for r in R for t in T), name="resource_capacity")
    m3_model.addConstrs((gp.quicksum(x[i,j,k,t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")

    for i in I:
        for k in K:
            for t in T:
                m3_model.addConstr(
                    D_tran[i,k,t] == gp.quicksum(beta[i,k,t] * lmbda[i,k,t] * x[i,j,k,t] * (h[k]+f[k]) / max(0.001,B[i,j]) for j in J),
                    name=f"trans_delay_def")

    m3_model.addConstrs((D_prop[i,k,t] == gp.quicksum(d[i,j] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="prop_delay_def")
    m3_model.addConstrs((D_proc[i,k,t] == gp.quicksum(v[j,k] * (f[k]+h[k]) * lmbda[i,k,t] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="proc_delay_def")
    m3_model.addConstrs((D_tran[i,k,t] + D_prop[i,k,t] + D_proc[i,k,t] <=  Delta[i,k] for i in I for k in K for t in T), name="max_delay")

    # Gurobi solver parameters
    m3_model.setParam('OutputFlag', 0)
    m3_model.setParam('TimeLimit', 600)
    m3_model.setParam('MIPGap', 0.05)
    m3_model.setParam('FeasibilityTol', 1e-5)
    m3_model.setParam('NumericFocus', 3)

    # Optimize
    try:
        m3_model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return m3_model, None

    if m3_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        # Extract solution
        C_energy_list, C_carbon_list, woc_list, C_delay_list,C_tot_with_time = [], [], [], [], []
        for t in T:
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            woc_list.append(sum(woc[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i, k, t].X + D_prop[i, k, t].X + D_proc[i, k, t].X) for i in I for k in K))

        C_tot_with_time = C_energy_list + C_carbon_list + C_delay_list
        C_tot = sum(C_energy_list) + sum(C_carbon_list) + sum(C_delay_list)

        m3_sol = {
            'x': {(i,j,k,t): x[i,j,k,t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j,t): P_g[j,t].X for j in J for t in T},
            'P_c': {(j,t): P_c[j,t].X for j in J for t in T},
            'P_d': {(j,t): P_d[j,t].X for j in J for t in T},
            'D_tran': {(i,k,t): D_tran[i,k,t].X for i in I for k in K for t in T},
            'D_prop': {(i,k,t): D_prop[i,k,t].X for i in I for k in K for t in T},
            'D_proc': {(i,k,t): D_proc[i,k,t].X for i in I for k in K for t in T},
            'woc': {(j,t): woc[j,t].X for j in J for t in T},
            'woc_with_time': woc_list,
            'C_energy_with_time': C_energy_list,
            'C_carbon_with_time': C_carbon_list,
            'C_delay_with_time': C_delay_list,
            'm3_cost_time':C_tot_with_time,
            'm3_C_energy': sum(C_energy_list),
            'm3_C_carbon': sum(C_carbon_list),
            'm3_C_delay': sum(C_delay_list),
            'm3_cost': C_tot,
            'm3_obj': m3_model.objVal,
            'status': m3_model.status
        }
        return m3_model, m3_sol
    else:
        print(f"Optimization failed: Status {m3_model.status}")
        if m3_model.status == GRB.INFEASIBLE:
            print("Computing IIS...")
            m3_model.computeIIS()
            m3_model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        return m3_model, None
    




def m4_model(params):
    """
    Implement the baseline model, minimizing the C1_cost only, energy-saving model

    Args:
        params: Dictionary containing all necessary parameters

    Returns:
        model: Solved Gurobi model
        solution: Dictionary containing optimal decision variables
    """
    # Extract parameters
    I, J, K, T, R = params['I'], params['J'], params['K'], params['T'], params['R']
    lmbda, h, f = params['lambda'], params['h'], params['f']
    tau_in, tau_out, c, theta, delta = params['tau_in'], params['tau_out'], params['c'], params['theta'], params['delta']
    P_w, P_max, PUE = params['P_w'], params['P_max'], params['PUE']
    C, alpha = params['C'], params['alpha']
    beta, B, d, v, Delta = params['beta'], params['B'], params['d'], params['v'], params['Delta']
    WUE, EWIF, Z = params['WUE'], params['EWIF'], params['Z']
    rho = params['rho']
    m4_model = gp.Model("Delay_min_model")

    # Variables
    x = m4_model.addVars(I, J, K, T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    P_g = m4_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_g")
    D_tran = m4_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
    D_prop = m4_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
    D_proc = m4_model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
    woc = m4_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="woc")
    eta = m4_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    P_d = m4_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_d")
    P_c = m4_model.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, name="P_c")

    m4_model.update()
    # Objective: minimize total delay
    C_delay = gp.quicksum(rho[k] * (D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t]) for i in I for k in K for t in T)
    C_energy = gp.quicksum(c[j,t] * P_g[j,t] for j in J for t in T)
    C_carbon = gp.quicksum(delta[j] * theta[j,t] * P_g[j,t] for j in J for t in T)
    m4_obj = 1/3*C_energy + 1/3*C_carbon + 1/3*C_delay
    m4_model.setObjective(m4_obj, GRB.MINIMIZE)

    # Constraints
    m4_model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) for j in J for t in T), name="computation_consumption")
    m4_model.addConstrs((P_d[j,t] == PUE[j] * P_c[j,t] for j in J for t in T), name="tot_power_consumption")
    m4_model.addConstrs((P_d[j,t] <= P_g[j,t] + P_w[j,t] for j in J for t in T), name="power_balance")
    m4_model.addConstrs((P_g[j,t] <= P_max[j,t] for j in J for t in T), name="grid_capacity")
    m4_model.addConstrs((woc[j,t] == (WUE[j,t] / PUE[j] + EWIF[j,t]) * P_d[j,t] for j in J for t in T), name="water_consumption")
    m4_model.addConstr(gp.quicksum(woc[j,t] for j in J for t in T) <= Z, name="long_term_water_footprint")
    m4_model.addConstrs((gp.quicksum(alpha[k,r] * (f[k] + h[k]) * lmbda[i,k,t] * x[i,j,k,t] for i in I for k in K) <= C[r,j] for j in J for r in R for t in T), name="resource_capacity")
    m4_model.addConstrs((gp.quicksum(x[i,j,k,t] for j in J) == 1 for i in I for k in K for t in T), name="workload_allocation")

    for i in I:
        for k in K:
            for t in T:
                m4_model.addConstr(
                    D_tran[i,k,t] == gp.quicksum(beta[i,k,t] * lmbda[i,k,t] * x[i,j,k,t] * (h[k]+f[k]) / max(0.001,B[i,j]) for j in J),
                    name=f"trans_delay_def_{i}_{k}_{t}"
                )

    m4_model.addConstrs((D_prop[i,k,t] == gp.quicksum(d[i,j] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="prop_delay_def")
    m4_model.addConstrs((D_proc[i,k,t] == gp.quicksum(v[j,k] * (f[k]+h[k]) * lmbda[i,k,t] * x[i,j,k,t] for j in J) for i in I for k in K for t in T), name="proc_delay_def")
    m4_model.addConstrs((D_tran[i,k,t] + D_prop[i,k,t] + D_proc[i,k,t] <= Delta[i,k] for i in I for k in K for t in T), name="max_delay")

    # Gurobi solver parameters
    m4_model.setParam('OutputFlag', 0)
    m4_model.setParam('TimeLimit', 600)
    m4_model.setParam('MIPGap', 0.05)
    m4_model.setParam('FeasibilityTol', 1e-5)
    m4_model.setParam('NumericFocus', 3)

    # Optimize
    try:
        m4_model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return m4_model, None

    if m4_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        # Extract solution
        C_energy_list, C_carbon_list, woc_list,C_delay_list = [], [], [], []
        for t in T:
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i, k, t].X + D_prop[i, k, t].X + D_proc[i, k, t].X) for i in I for k in K))
            woc_list.append(sum(woc[j,t].X for j in J))
        
        C_tot = sum(C_energy_list) + sum(C_carbon_list) + sum(C_delay_list)

        m4_sol = {
            'x': {(i,j,k,t): x[i,j,k,t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j,t): P_g[j,t].X for j in J for t in T},
            'P_c': {(j,t): P_c[j,t].X for j in J for t in T},
            'P_d': {(j,t): P_d[j,t].X for j in J for t in T},
            'D_tran': {(i,k,t): D_tran[i,k,t].X for i in I for k in K for t in T},
            'D_prop': {(i,k,t): D_prop[i,k,t].X for i in I for k in K for t in T},
            'D_proc': {(i,k,t): D_proc[i,k,t].X for i in I for k in K for t in T},
            'woc': {(j,t): woc[j,t].X for j in J for t in T},
            'woc_with_time': woc_list,
            'C_energy_with_time': C_energy_list,
            'C_carbon_with_time': C_carbon_list,
            'C_delay_with_time': C_delay_list,
            'm4_C_energy': sum(C_energy_list),
            'm4_C_carbon': sum(C_carbon_list),
            'm4_C_delay': sum(C_delay_list),
            'm4_cost': C_tot,
            'm4_obj': m4_model.objVal,
            'status': m4_model.status
        }
        return m4_model, m4_sol
    else:
        print(f"Optimization failed: Status {m4_model.status}")
        if m4_model.status == GRB.INFEASIBLE:
            print("Computing IIS...")
            m4_model.computeIIS()
            m4_model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        return m4_model, None