import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations

def lexicographic_green_llm(params, priority_order=['carbon', 'delay', 'energy'], tolerance=0.001):
    """
    Implement lexicographic optimization for Green-LLM
    
    Solves objectives sequentially in order of priority:
    1. Optimize highest priority objective
    2. Add constraint to maintain this optimum (with tolerance)
    3. Optimize next priority objective subject to this constraint
    4. Repeat until all objectives are optimized
    
    Args:
        params: Dictionary containing all necessary parameters
        priority_order: List of objectives in order of priority ['carbon', 'energy', 'delay']
        tolerance: Tolerance for maintaining previous optimal values (as fraction)
        
    Returns:
        model: Final solved Gurobi model
        solution: Dictionary containing optimal decision variables
        phase_results: List of results from each optimization phase
    """
    print(f"Priority order: {' > '.join(priority_order)}")
    print("-" * 50)
    
    # Validate priority order
    valid_objectives = {'energy', 'carbon', 'delay'}
    if not all(obj in valid_objectives for obj in priority_order):
        raise ValueError(f"Invalid objectives. Must be from {valid_objectives}")
    if len(set(priority_order)) != 3:
        raise ValueError("Must specify all three objectives exactly once")
    
    phase_results = []
    optimal_values = {}
    
    # Extract parameters (same as other methods)
    I, J, K, T, R = params['I'], params['J'], params['K'], params['T'], params['R']
    lmbda, h, f, rho = params['lambda'], params['h'], params['f'], params['rho']
    tau_in, tau_out, c, theta, delta = params['tau_in'], params['tau_out'], params['c'], params['theta'], params['delta']
    P_w, P_max, PUE = params['P_w'], params['P_max'], params['PUE']
    C, alpha = params['C'], params['alpha']
    beta, B, d, v, Delta = params['beta'], params['B'], params['d'], params['v'], params['Delta']
    WUE, EWIF, Z = params['WUE'], params['EWIF'], params['Z']
    
    for phase, objective in enumerate(priority_order):
        print(f"\nPhase {phase + 1}: Optimizing {objective.upper()}")
        
        # Create model for this phase
        model = gp.Model(f"Lexicographic_Phase_{phase+1}_{objective}")
        
        # Decision variables
        x = model.addVars([(i, j, k, t) for i in I for j in J for k in K for t in T], 
                          vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        P_g = model.addVars([(j, t) for j in J for t in T], 
                            vtype=GRB.CONTINUOUS, lb=0, name="P_g")
        D_tran = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                               vtype=GRB.CONTINUOUS, lb=0, name="D_tran")
        D_prop = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                               vtype=GRB.CONTINUOUS, lb=0, name="D_prop")
        D_proc = model.addVars([(i, k, t) for i in I for k in K for t in T], 
                               vtype=GRB.CONTINUOUS, lb=0, name="D_proc")
        woc = model.addVars([(j, t) for j in J for t in T], 
                            vtype=GRB.CONTINUOUS, lb=0, name="woc")
        eta = model.addVars([(j, t) for j in J for t in T], 
                            vtype=GRB.CONTINUOUS, lb=0, name="eta")
        P_d = model.addVars([(j, t) for j in J for t in T], 
                            vtype=GRB.CONTINUOUS, lb=0, name="P_d")
        P_c = model.addVars([(j,t) for j in J for t in T], 
                            vtype=GRB.CONTINUOUS, lb=0, name="P_c")
        
        model.update()
        
        # Define objective functions
        C_energy = gp.quicksum(c[j, t] * P_g[j, t] for j in J for t in T)
        C_carbon = gp.quicksum(delta[j] * theta[j,t] * P_g[j, t] for j in J for t in T)
        C_delay = gp.quicksum(rho[k] * (D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t]) 
                             for i in I for k in K for t in T)
        
        # Set objective for current phase
        if objective == 'energy':
            model.setObjective(C_energy, GRB.MINIMIZE)
        elif objective == 'carbon':
            model.setObjective(C_carbon, GRB.MINIMIZE)
        elif objective == 'delay':
            model.setObjective(C_delay, GRB.MINIMIZE)
        
        # Add constraints from previous phases
        for prev_obj, optimal_val in optimal_values.items():
            constraint_bound = optimal_val * (1 + tolerance)
            print(f"   Adding constraint: {prev_obj} ≤ {constraint_bound:.2f}")
            
            if prev_obj == 'energy':
                model.addConstr(C_energy <= constraint_bound, f"maintain_{prev_obj}_optimum")
            elif prev_obj == 'carbon':
                model.addConstr(C_carbon <= constraint_bound, f"maintain_{prev_obj}_optimum")
            elif prev_obj == 'delay':
                model.addConstr(C_delay <= constraint_bound, f"maintain_{prev_obj}_optimum")
        
        # Add all original constraints (same as other methods)
        # Computational energy consumption
        model.addConstrs((P_c[j,t] == gp.quicksum((tau_in[k] * h[k] + tau_out[k] * f[k]) * lmbda[i, k, t] * x[i, j, k, t] 
                                                  for i in I for k in K) for j in J for t in T), 
                         name="computation_consumption")
        
        # Total Energy consumption
        model.addConstrs((P_d[j, t] == PUE[j] * P_c[j, t] for j in J for t in T), 
                         name="tot_power_consumption")
        
        # Power balance constraints
        model.addConstrs((P_d[j, t] <= P_g[j, t] + P_w[j, t] for j in J for t in T), 
                         name="power_balance")
        
        # Grid capacity condition
        model.addConstrs((P_g[j, t] <= P_max[j, t] for j in J for t in T), 
                         name="grid_capacity")
        
        # Water consumption constraints
        model.addConstrs((woc[j, t] == (WUE[j, t] / PUE[j] + EWIF[j, t]) * P_d[j, t] for j in J for t in T),
                         name="water_consumption")
        model.addConstr(gp.quicksum(woc[j, t] for j in J for t in T) <= Z, 
                        name="long_term_water_footprint")
        
        # Resource capacity constraints
        model.addConstrs((gp.quicksum(alpha[k, r] * (f[k]+h[k]) * lmbda[i, k, t] * x[i, j, k, t] 
                                      for i in I for k in K) <= C[r, j] 
                          for j in J for r in R for t in T),
                         name="resource_capacity")
        
        # Workload allocation constraints
        model.addConstrs((gp.quicksum(x[i, j, k, t] for j in J) == 1 for i in I for k in K for t in T), 
                         name="workload_allocation")
        
        # Delay constraints
        for i in I:
            for k in K:
                for t in T:
                    model.addConstr(
                        D_tran[i, k, t] == gp.quicksum(beta[i, k, t] * lmbda[i, k, t] * x[i, j, k, t] * (h[k] + f[k]) / 
                                                       max(0.001, B[i, j]) for j in J),
                        f"trans_delay_def_{i}_{k}_{t}"
                    )
        
        model.addConstrs((D_prop[i, k, t] == gp.quicksum(d[i, j] * x[i, j, k, t] for j in J) 
                          for i in I for k in K for t in T), name="prop_delay_def")
        
        model.addConstrs((D_proc[i, k, t] == gp.quicksum(v[j, k] * (f[k]+h[k])* lmbda[i,k,t] * x[i, j, k, t] for j in J) 
                          for i in I for k in K for t in T), name="proc_delay_def")
        
        model.addConstrs((D_tran[i, k, t] + D_prop[i, k, t] + D_proc[i, k, t] <= Delta[i, k] 
                          for i in I for k in K for t in T), name="max_delay")
        
        # Optimization parameters
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 600)
        model.setParam('MIPGap', 0.05)
        model.setParam('FeasibilityTol', 1e-5)
        model.setParam('NumericFocus', 3)
        
        # Optimize
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"Optimization error in phase {phase + 1}: {e}")
            return None, None, phase_results
        
        if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            print(f"Phase {phase + 1} failed with status {model.status}")
            return None, None, phase_results
        
        # Extract results for this phase
        current_energy = C_energy.getValue()
        current_carbon = C_carbon.getValue()
        current_delay = C_delay.getValue()
        
        # Store optimal value for this objective
        if objective == 'energy':
            optimal_values['energy'] = current_energy
        elif objective == 'carbon':
            optimal_values['carbon'] = current_carbon
        elif objective == 'delay':
            optimal_values['delay'] = current_delay
        
        phase_result = {
            'phase': phase + 1,
            'objective': objective,
            'optimal_value': model.objVal,
            'energy_cost': current_energy,
            'carbon_cost': current_carbon,
            'delay_cost': current_delay,
            'total_cost': current_energy + current_carbon + current_delay,
            'status': model.status
        }
        
        phase_results.append(phase_result)
        
        print(f"   Optimal {objective}: {model.objVal:.2f}")
        print(f"   Current costs - Energy: {current_energy:.2f}, Carbon: {current_carbon:.2f}, Delay: {current_delay:.2f}")
    
    # Final solution extraction
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        C_energy_list, C_carbon_list, C_delay_list, woc_list = [], [], [], []
        
        for t in T:
            C_energy_list.append(sum(c[j,t] * P_g[j,t].X for j in J))
            C_carbon_list.append(sum(delta[j] * theta[j,t] * P_g[j,t].X for j in J))
            C_delay_list.append(sum(rho[k] * (D_tran[i,k,t].X + D_prop[i,k,t].X + D_proc[i,k,t].X) 
                                   for i in I for k in K))
            woc_list.append(sum(woc[j,t].X for j in J))
        
        lexi_sol = {
            'x': {(i, j, k, t): x[i, j, k, t].X for i in I for j in J for k in K for t in T},
            'P_g': {(j, t): P_g[j, t].X for j in J for t in T},
            'P_c': {(j, t): P_c[j, t].X for j in J for t in T},
            'P_d': {(j, t): P_d[j, t].X for j in J for t in T},
            'D_tran': {(i, k, t): D_tran[i, k, t].X for i in I for k in K for t in T},
            'D_prop': {(i, k, t): D_prop[i, k, t].X for i in I for k in K for t in T},
            'D_proc': {(i, k, t): D_proc[i, k, t].X for i in I for k in K for t in T},
            'woc': {(j, t): woc[j, t].X for j in J for t in T},
            'C_energy_cost': C_energy.getValue(),
            'C_carbon_cost': C_carbon.getValue(),
            'C_delay_cost': C_delay.getValue(),
            'total_cost': C_energy.getValue() + C_carbon.getValue() + C_delay.getValue(),
            'priority_order': priority_order,
            'optimal_values': optimal_values,
            'status': model.status,
            'method': 'lexicographic'
        }
        
        
        return model, lexi_sol, phase_results
    else:
        return None, None, phase_results

