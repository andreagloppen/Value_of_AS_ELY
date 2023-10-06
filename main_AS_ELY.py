import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
# UNITS: MW, MWh and kg (H2)



########################################################################################################################
# Define run setting
########################################################################################################################

mipgap = 0.0001     # Solver parameter
timelimit = 5 * 60  # Solver parameter
case = 'DK2_22'     # DK1_22, DK1_21, DK2_22,
FCR_factor = 0      # Is the FCR reserve included? Yes or no
mFRR_factor = 1     # Is the mFRR reserve included? Yes or no

Days = 365          # Number of days in analysis

dir = 'Value_of_AS_ELY'                         # Location of 'Value_of_AS_ELY' folder
model = ['alpha_1', 'alpha_0', 'alpha_r', 'no_as']  # Model versions 'alpha_1', 'alpha_0', 'alpha_r' or 'no_as'


########################################################################################################################
# Define ranges
########################################################################################################################

dT = 1              # hourly resolution
NT = int(24/dT)     # number of time steps
ND = 5              # number of dispensers/tube trailers
NI = int(NT/4)      # number of 4-hour blocks in horizon
NJ = int(4)         # number of 1 hour intervals in 4 hours
N_down = int(2/dT)  # number of down timesteps after ely shut of
NS = 4              # number of segments

D = range(ND)       # range over dispensers/tube trailers
T = range(NT)       # range timesteps in horizon (24 hours)
I = range(NI)       # range over FCR blocks
J = range(NJ)       # range over mFRR blocks
S = range(NS)       # range over piecewise segments on ely production curve

T1 = range(1,NT)    # range over time excluding the first step


########################################################################################################################
# Define electrolyzer parameters
########################################################################################################################

C_E = 10    # Electrolyzer capacity in MW
E_min = 0.1 # Electrolyzer miniumum load in %

# Pieces on production curve h = A * p + B
A = [21.15664817, 18.96085366, 16.87083602, 14.99118166]
B = [-0.35819557,  0.19075305,  1.23576187,  2.64550265]

P_min = np.array([0.1 , 0.25001, 0.5001 , 0.75001]) * C_E   # Minimum load per electrolyzer segment in MW
P_max = np.array([0.25, 0.5 , 0.75, 1.  ]) * C_E            # Maximum load per electrolyzer segment in MW
P_sb = 0.025 * C_E                                          # Standby consumption electrolyzer in MW

S_max = 1000    # Maximum capacity of tube trailer in kg
H_min = 2000    # Minimum daily demand in kg
C_D = 1000      # Dispenser capacity in kg

TT_bin = np.zeros((NT,ND))+1 # Tube trailer schedule, put to always available for simplicity




########################################################################################################################
# Define time series import functions
########################################################################################################################


# Import the time-series for the full case
def import_full(case):
    """ Import price and activation time series.
    """
    df_full = pd.DataFrame()
    if case == 'DK1_21':
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/DA_DK1_2021.xlsx'), columns=['SpotPriceEUR'])
        df_full['pi_DA'] = df.iloc[:, 0]

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FCR_DK1_2021.xlsx'),
                          columns=['DK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'])
        df_full['pi_FCR'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2021.xlsx'), columns=['mFRR_DownPriceEUR'])
        df_full['pi_mFRR_dw'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2021.xlsx'), columns=['mFRR_UpPriceEUR'])
        df_full['pi_mFRR_up'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2021.xlsx'),
                          columns=['ImbalancePriceEUR'])
        df_full['pi_bal'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2021.xlsx'), columns=['Act_dw'])
        df_full['act_dw'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2021.xlsx'), columns=['Act_up'])
        df_full['act_up'] = df


        pi_H2 = 10


    if case == 'DK1_22':
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/DA_DK1_2022.xlsx'), columns=['SpotPriceEUR'])
        df_full['pi_DA'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FCR_DK1_2022.xlsx'),
                          columns=['DENMARK_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'])
        df_full['pi_FCR'] = df


        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2022.xlsx'), columns=['mFRR_DownPriceEUR'])
        df_full['pi_mFRR_dw'] = df


        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2022.xlsx'), columns=['mFRR_UpPriceEUR'])
        df_full['pi_mFRR_up'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2022.xlsx'),
                          columns=['ImbalancePriceEUR'])
        df_full['pi_bal'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2022.xlsx'), columns=['Act_dw'])
        df_full['act_dw'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2022.xlsx'), columns=['Act_up'])
        df_full['act_up'] = df

        pi_H2 = 10

    if case == 'DK2_22':
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/DA_DK2_2022.xlsx'), columns=['SpotPriceEUR'])
        df_full['pi_DA'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FCR_DK2_2022.xlsx'), columns=['FCR_N_PriceEUR'])
        df_full['pi_FCR'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK2_2022.xlsx'), columns=['mFRR_DownPriceEUR'])
        df_full['pi_mFRR_dw'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK2_2022.xlsx'), columns=['mFRR_UpPriceEUR'])
        df_full['pi_mFRR_up'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK2_2022.xlsx'),
                          columns=['ImbalancePriceEUR'])
        df_full['pi_bal'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK2_2022.xlsx'), columns=['Act_dw'])
        df_full['act_dw'] = df

        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK2_2022.xlsx'), columns=['Act_up'])
        df_full['act_up'] = df

        pi_H2 = 10

    # Specify that we only assume histroical activation if our price bid is low enough
    for i in range(len(df_full['pi_bal'])):
        if df_full['pi_bal'].iloc[i] < 19 / pi_H2:
            df_full['act_dw'].iloc[i] = 0
            df_full['act_up'].iloc[i] = 0

    return pi_H2, df_full

# Import the time-series for the day d
def import_daily(df_full, d, Days):
    """ Outputs time-series for the current day d only.
    """
    pi_DA_mat = np.split(df_full['pi_DA'].to_numpy(), Days)
    pi_FCR_mat = np.split(df_full['pi_FCR'].to_numpy(), Days)
    pi_mFRR_dw_mat = np.split(df_full['pi_mFRR_dw'].to_numpy(), Days)
    pi_mFRR_up_mat = np.split(df_full['pi_mFRR_up'].to_numpy(), Days)
    pi_bal_mat = np.split(df_full['pi_bal'].to_numpy(), Days)

    act_dw_mat = np.split(df_full['act_dw'].to_numpy(), Days)
    act_up_mat = np.split(df_full['act_up'].to_numpy(), Days)


    pi_DA = pi_DA_mat[d].flatten()
    pi_DA[np.isnan(pi_DA)] = 0

    pi_FCR = pi_FCR_mat[d].flatten()
    pi_FCR[np.isnan(pi_FCR)] = 0

    pi_mFRR_dw = pi_mFRR_dw_mat[d].flatten()
    pi_mFRR_dw[np.isnan(pi_mFRR_dw)] = 0

    pi_mFRR_up = pi_mFRR_up_mat[d].flatten()
    pi_mFRR_up[np.isnan(pi_mFRR_up)] = 0

    pi_bal = pi_bal_mat[d].flatten()
    pi_bal[np.isnan(pi_bal)] = 0

    act_dw = act_dw_mat[d].flatten()
    act_dw[np.isnan(act_dw)] = 0

    act_up = act_up_mat[d].flatten()
    act_up[np.isnan(act_up)] = 0

    return pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_dw, act_up




########################################################################################################################
# Define optimization model in functions
########################################################################################################################

#Model formulation to place DA and reserve bids based on assumption on activation 'alpha'
def AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw, alpha_up):

    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("ELY_AS")

        p_E = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_E")  # electrolyzer power [MW]
        p_sb = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb")  # electrolyzer sb power [MW]
        p_tot = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot")  # electrolyzer power over segments [MW]

        p_act_dw = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_dw")  # dw act. power [MW]
        p_act_up = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_up")  # up act. power [MW]

        z_E = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_E")  # electrolyzer segment
        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act_dw = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_dw")  # electrolyzer segment
        z_act_up = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_up")  # electrolyzer segment


        # HYDROGEN VARIABLES
        h_E = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_E")  # expected hydrogen prod (no act) [kg]
        h_act_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act_up")  # expected up-activated hydrogen prod [kg]
        h_act_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act_dw")  # expected dw-activated hydrogen prod [kg]
        h_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_D_dw")  # expected dw-activated tube-trailer hydrogen input [kg]
        s_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="s_D_dw")  # expected dw-activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # Set objective
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       , GRB.MINIMIZE) #(1)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act_up[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I)  * NJ)
        m.addConstr(rev_mFRR_dw == qsum(bid_mFRR_dw[t] * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum(bid_mFRR_up[t] * pi_mFRR_up[t] for t in T) * dT)

        # ---------------- Electrolyzer constraints ----------------

        m.addConstrs(p_E[t,s] <= P_max[s] * z_E[t,s]  for s in S for t in T) # (2)
        m.addConstrs(P_min[s] * z_E[t,s] <= p_E[t,s]  for s in S for t in T) # (2)
        m.addConstrs(qsum(z_E[t,s] for s in S) <= 1  for t in T)      # (3)

        m.addConstrs(h_E[t] == qsum(A[s] * p_E[t,s] + B[s] * z_E[t,s] for s in S) * dT for t in T) # (4)

        m.addConstrs(p_tot[t] == qsum(p_E[t,s] for s in S) for t in T) # (5)



        m.addConstrs(z_on[t] == qsum(z_E[t,s] for s in S) for t in T) # (6)

        m.addConstrs(p_sb[t] == z_sb[t] * P_sb for t in T) # (7)

        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (8)

        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (9)


        # ---------------- Market-bids constraints ----------------

        # DA bid
        m.addConstrs(bid_DA[t] == dT * (p_tot[t]  + p_sb[t]) for t in T) # (10)

        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J) # (11)
        m.addConstrs(bid_FCR[i] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J) # (12)

        # mFRR down and up bids
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - p_tot[t] for t in T) # (13)
        m.addConstrs(bid_mFRR_up[t] <= p_tot[t] - C_E * E_min * (1 - z_off[t]) for t in T) # (14)

        # Sum of reserve bids
        m.addConstrs(bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J) # (15)
        m.addConstrs(bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J) # (16)

        # Bid size limitations
        m.addConstrs(FCR_min*z_FCR[i] <= bid_FCR[i] for i in I) # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max*z_FCR[i] for i in I) # (17)

        m.addConstrs(mFRR_dw_min*z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T) # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max*z_mFRR_dw[t] for t in T) # (18)

        m.addConstrs(mFRR_up_min*z_mFRR_up[t] <= bid_mFRR_up[t] for t in T) # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max*z_mFRR_up[t] for t in T) # (19)

        # Up-activation constraints
        m.addConstrs(qsum(p_act_up[t,s] for s in S) == p_tot[t] - bid_mFRR_up[t]*alpha_up[t] for t in T) # (20)

        m.addConstrs(p_act_up[t,s] <= P_max[s] * z_act_up[t,s]  for s in S for t in T) # (21)
        m.addConstrs(P_min[s] * z_act_up[t,s] <= p_act_up[t,s]  for s in S for t in T) # (21)
        m.addConstrs(qsum(z_act_up[t,s] for s in S ) == 1 for t in T) # (22)

        m.addConstrs(h_act_up[t] == qsum(A[s] * p_act_up[t,s] + B[s] * z_act_up[t,s] for s in S) * dT for t in T) # (23)

        m.addConstr(H_min <= qsum(h_act_up[t]  for t in T)) # (24)

        # Dw-activation constraints
        m.addConstrs(qsum(p_act_dw[t,s] for s in S) == p_tot[t] + bid_mFRR_dw[t]*alpha_dw[t] for t in T) # (25)

        m.addConstrs(p_act_dw[t,s] <= P_max[s] * z_act_dw[t,s]  for s in S for t in T) # (26)
        m.addConstrs(P_min[s] * z_act_dw[t,s] <= p_act_dw[t,s]  for s in S for t in T) # (26)
        m.addConstrs(qsum(z_act_dw[t,s] for s in S ) == 1 for t in T) # (27)


        m.addConstrs(h_act_up[t] == qsum(A[s] * p_act_up[t,s] + B[s] * z_act_up[t,s] for s in S) * dT for t in T) # (23)

        m.addConstrs(h_act_dw[t] == qsum(A[s] * p_act_dw[t,s] + B[s] * z_act_dw[t,s] for s in S) * dT for t in T) # (28)

        m.addConstrs(h_act_dw[t] == qsum(h_D_dw[t, d] for d in D) for t in T)  # (29)
        m.addConstrs(h_D_dw[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  # (30)
        m.addConstrs(s_D_dw[0, d] == h_D_dw[0, d] for d in D)  # (31)
        m.addConstrs(s_D_dw[t, d] == h_D_dw[t, d] + s_D_dw[t - 1, d] for t in T1 for d in D)  # (32)
        m.addConstrs(s_D_dw[t, d] <= S_max for d in D for t in T) # (33)

        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_tot', 'p_sb', 'h_E', 'z_on', 'z_sb', 'z_off', 'bid_mFRR_up', 'h_act_up', 'bid_mFRR_dw', 'h_act_dw', 'bid_DA']

        data = pd.DataFrame(0, index=np.arange(NT), columns=var_names_t)
        for i in var_names_t:
            data[i] = [var.x for var in m.getVars() if i in var.VarName]

        bid_FCR = [bid_FCR[i].x for i in I]
        data = data.assign(bid_FCR = np.repeat(bid_FCR,4))
        data = data.assign(alpha_up = alpha_up)
        data = data.assign(alpha_dw = alpha_dw)

        data['objective'] = m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))


    return data

# Perfect foresight on activation aka 'oracle'
def AS_ELY_perfect(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up, act_dw):

    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("REALIZATION")


        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_tot_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot_act")  # act. power [MW]
        p_sb_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb_act")  # electrolyzer sb power [MW]

        p_bal = m.addVars(NT,  lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # imbalance [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer activated segment

        slack_dw = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_dw")  # slack on activation [MW]
        slack_up = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_up")  # slack on activation [MW]

        # HYDROGEN VARIABLES
        h_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act")  # activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_D")  # activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="s_D")  # activated tube-trailer storage state [kg]


        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state


        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_bal")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # ---------------- Set Objective ----------------
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw * dT  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up * dT  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal # Revenue from delta energy (can be both positive [dw-reg] and negative [up-reg])
                       + 100000 * qsum(slack_up[t] + slack_dw[t] for t in T)
                       , GRB.MINIMIZE)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NI  )
        m.addConstr(rev_mFRR_dw == qsum( (bid_mFRR_dw[t]- slack_dw[t]) * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum( (bid_mFRR_up[t] - slack_up[t])* pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # ---------------- Activation constraints ----------------
        m.addConstrs(p_bal[t] == bid_mFRR_dw[t] * act_dw[t] - slack_dw[t] - bid_mFRR_up[t] * act_up[t] + slack_up[t] for t in T)
        m.addConstrs(qsum(p_act[t,s] for s in S) == bid_DA[t] + p_bal[t] for t in T)
        m.addConstrs(p_act[t, s] <= P_max[s] * z_act[t, s] for s in S for t in T)  #
        m.addConstrs(P_min[s] * z_act[t, s] <= p_act[t, s] for s in S for t in T)  #
        m.addConstrs(qsum(z_act[t, s] for s in S) <= 1 for t in T)  # (3)

        m.addConstrs(p_tot_act[t] == qsum(p_act[t, s] for s in S) for t in T)  # (2)

        m.addConstrs(z_on[t] == qsum(z_act[t, s] for s in S) for t in T)  # (5)
        m.addConstrs(p_sb_act[t] == P_sb * z_sb[t] for t in T)  # (6)
        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (7)
        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (8)

        # ---------------- Hydrogen constraints ----------------
        m.addConstrs(h_act[t] == qsum(A[s] * p_act[t, s] + B[s] * z_act[t, s] for s in S) * dT for t in T)

        m.addConstr(H_min <= qsum(h_act[t] for t in T))

        m.addConstrs(h_act[t] == qsum(h_D[t, d] for d in D) for t in T)  #
        m.addConstrs(h_D[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  #
        m.addConstrs(s_D[0, d] == h_D[0, d] for d in D)  #
        m.addConstrs(s_D[t, d] == h_D[t, d] + s_D[t - 1, d] for t in T1 for d in D)  #
        m.addConstrs(s_D[t, d] <= S_max for d in D for t in T)

        # ---------------- Bid constraints ----------------
        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - bid_DA[i * NJ + j] for i in I for j in J)  # (11)
        m.addConstrs(
            bid_FCR[i] <= bid_DA[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J)  # (12)

        # mFRR down and up bids
        # TODO: Should FCR be limited to only occur when there is an on-state, while mFRR can occur both from on and sb?
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - bid_DA[t] for t in T)  # (13)
        m.addConstrs(bid_mFRR_up[t] <= bid_DA[t] - C_E * E_min * (1 - z_off[t]) for t in T)  # (14)

        # Sum of reserve bids
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - bid_DA[i * NJ + j] for i in I for j
            in J)  # (15)
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= bid_DA[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I
            for j in J)  # (16)

        # Bid size limitations
        m.addConstrs(FCR_min * z_FCR[i] <= bid_FCR[i] for i in I)  # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max * z_FCR[i] for i in I)  # (17)

        m.addConstrs(mFRR_dw_min * z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T)  # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max * z_mFRR_dw[t] for t in T)  # (18)

        m.addConstrs(mFRR_up_min * z_mFRR_up[t] <= bid_mFRR_up[t] for t in T)  # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max * z_mFRR_up[t] for t in T)  # (19)

        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_tot_act', 'p_sb_act', 'h_act', 'slack_dw', 'slack_up']
        # , 'mFRR_up_C_rev', 'mFRR_dw_C_rev', 'mFRR_up_A_rev', 'mFRR_dw_A_rev']

        real = pd.DataFrame()

        real['objective'] = pd.Series(m.objVal).repeat(NT).reset_index(drop=True) - 100000 * sum(slack_up[i].x + slack_dw[i].x  for i in range(NT))

        real['p_tot'] = pd.Series(0).repeat(NT).reset_index(drop=True)
        real['p_sb'] = pd.Series(0).repeat(NT).reset_index(drop=True)
        real['h_E'] = pd.Series(0).repeat(NT).reset_index(drop=True)

        for i in var_names_t:
            real[i] = [var.x for var in m.getVars() if i in var.VarName]

        p_acts = np.zeros((NT, NS))
        for s in S:
            column = 'p_act' + str(s)
            for t in T:
                p_acts[t, s] = p_act[t, s].x
            real[column] = p_acts[:, s]


        real = real.assign(act_up = act_up)
        real = real.assign(act_dw = act_dw)

        rev_var_names = ['rev_DA', 'rev_H2', 'rev_FCR', 'rev_mFRR_dw', 'rev_mFRR_up', 'rev_bal']

        revenues = pd.DataFrame(columns=rev_var_names)
        for i in rev_var_names:
            revenues[i] =  [var.x for var in m.getVars() if i in var.VarName]

        real['bid_DA'] = pd.DataFrame(bid_DA[t].x for t in T)
        real['pi_DA'] = pd.DataFrame(pi_DA)
        real['bid_mFRR_up'] =  pd.DataFrame(bid_mFRR_up[t].x for t in T)
        real['pi_mFRR_up'] = pd.DataFrame(pi_mFRR_up)
        real['bid_mFRR_dw'] =  pd.DataFrame(bid_mFRR_dw[t].x for t in T)
        real['pi_mFRR_dw'] = pd.DataFrame(pi_mFRR_dw)
        real['bid_FCR'] =  pd.Series(bid_FCR[i].x for i in I).repeat(NI).reset_index(drop=True)
        real['pi_FCR'] = pd.DataFrame(pi_FCR)

        pwr_bal = np.zeros(NT)
        for t in T:
            pwr_bal[t] = p_bal[t].x
        real['pwr_bal'] = pd.DataFrame(pwr_bal)

        real['pi_bal'] = pd.DataFrame(pi_bal)


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return real, revenues

# Realization of activation (slack variable on the service provision)
def realization_AS_slack(data, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up, act_dw):

    bid_FCR = data.bid_FCR[::4].reset_index(drop = True)
    penalty = 100000

    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("REALIZATION")


        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_tot_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot_act")  # act. power [MW]
        p_sb_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb_act")  # electrolyzer sb power [MW]

        p_bal = m.addVars(NT,  lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # imbalance [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer activated segment

        slack_dw = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_dw")  # slack on activation [MW]
        slack_up = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_up")  # slack on activation [MW]

        # HYDROGEN VARIABLES
        h_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act")  # activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_D")  # activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="s_D")  # activated tube-trailer storage state [kg]

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_bal")  # mFRR up reserve revenue [DKK]

        # Solver settings
        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # ---------------- Set Objective ----------------
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw * dT  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up * dT  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal # Revenue from delta energy (can be both positive [dw-reg] and negative [up-reg])
                       + penalty * qsum(slack_up[t] + slack_dw[t] for t in T)
                       , GRB.MINIMIZE)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(data.bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ  )
        m.addConstr(rev_mFRR_dw == qsum( (data.bid_mFRR_dw[t]- slack_dw[t]) * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum( (data.bid_mFRR_up[t] - slack_up[t])* pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # ---------------- Activation constraints ----------------
        m.addConstrs(p_bal[t] == data.bid_mFRR_dw[t] * act_dw[t] - slack_dw[t] - data.bid_mFRR_up[t] * act_up[t] + slack_up[t] for t in T)
        m.addConstrs(qsum(p_act[t,s] for s in S) == data.p_tot[t] + p_bal[t] for t in T)
        m.addConstrs(p_act[t, s] <= P_max[s] * z_act[t, s] for s in S for t in T)
        m.addConstrs(P_min[s] * z_act[t, s] <= p_act[t, s] for s in S for t in T)
        m.addConstrs(qsum(z_act[t, s] for s in S) <= 1 for t in T)

        m.addConstrs(p_tot_act[t] == qsum(p_act[t, s] for s in S) for t in T)

        m.addConstrs(z_on[t] == qsum(z_act[t, s] for s in S) for t in T)
        m.addConstrs(p_sb_act[t] == P_sb * z_sb[t] for t in T)
        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))
        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)

        # ---------------- Hydrogen constraints ----------------
        m.addConstrs(h_act[t] == qsum(A[s] * p_act[t, s] + B[s] * z_act[t, s] for s in S) * dT for t in T)

        m.addConstr(H_min <= qsum(h_act[t] for t in T))

        m.addConstrs(h_act[t] == qsum(h_D[t, d] for d in D) for t in T)
        m.addConstrs(h_D[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)
        m.addConstrs(s_D[0, d] == h_D[0, d] for d in D)
        m.addConstrs(s_D[t, d] == h_D[t, d] + s_D[t - 1, d] for t in T1 for d in D)
        m.addConstrs(s_D[t, d] <= S_max for d in D for t in T)

        m.optimize()

        print('Obj: %g' % m.objVal)


        # Save results in dataframe
        var_names_t = ['p_tot_act', 'p_sb_act', 'h_act', 'slack_dw', 'slack_up']

        real = pd.DataFrame()

        real['objective'] = pd.Series(m.objVal).repeat(NT).reset_index(drop=True) - penalty * sum(slack_up[i].x + slack_dw[i].x  for i in range(NT))

        real['p_tot'] = data.p_tot
        real['p_sb'] = data.p_sb
        real['h_E'] = data.h_E

        for i in var_names_t:
            real[i] = [var.x for var in m.getVars() if i in var.VarName]

        p_acts = np.zeros((NT, NS))
        for s in S:
            column = 'p_act' + str(s)
            for t in T:
                p_acts[t, s] = p_act[t, s].x
            real[column] = p_acts[:, s]


        real = real.assign(act_up = act_up)
        real = real.assign(act_dw = act_dw)

        rev_var_names = ['rev_DA', 'rev_H2', 'rev_FCR', 'rev_mFRR_dw', 'rev_mFRR_up', 'rev_bal']

        revenues = pd.DataFrame(columns=rev_var_names)
        for i in rev_var_names:
            revenues[i] =  [var.x for var in m.getVars() if i in var.VarName]

        real['bid_DA'] = data.bid_DA
        real['pi_DA'] = pd.DataFrame(pi_DA)
        real['bid_mFRR_up'] = data.bid_mFRR_up
        real['pi_mFRR_up'] = pd.DataFrame(pi_mFRR_up)
        real['bid_mFRR_dw'] = data.bid_mFRR_dw
        real['pi_mFRR_dw'] = pd.DataFrame(pi_mFRR_dw)
        real['bid_FCR'] = data.bid_FCR
        real['pi_FCR'] = pd.DataFrame(pi_FCR)

        pwr_bal = np.zeros(NT)
        for t in T:
            pwr_bal[t] = p_bal[t].x
        real['pwr_bal'] = pd.DataFrame(pwr_bal)

        real['pi_bal'] = pd.DataFrame(pi_bal)


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return real, revenues

# Realization of activation (slack variable on the daily demand)
def realization_h2_slack(data, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up, act_dw):

    bid_FCR = data.bid_FCR[::4].reset_index(drop = True)
    penalty = 100000


    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("REALIZATION")


        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_tot_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot_act")  # act. power [MW]
        p_sb_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb_act")  # electrolyzer sb power [MW]

        p_bal = m.addVars(NT,  lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # imbalance [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer activated segment

        slack_dw = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_dw")  # slack on activation [MW]
        slack_up = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_up")  # slack on activation [MW]

        slack_aux = m.addVars(NT,  lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="slack_aux")  # slack on activation [MW]

        # HYDROGEN VARIABLES
        h_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act")  # activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_D")  # activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="s_D")  # activated tube-trailer storage state [kg]

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_bal")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # ---------------- Set Objective ----------------
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw * dT  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up * dT  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal # Revenue from delta energy (can be both positive [dw-reg] and negative [up-reg])
                       + penalty * qsum(slack_up[t] + slack_dw[t] for t in T)
                       + penalty * penalty * qsum(slack_aux[t]  for t in T)
                       , GRB.MINIMIZE)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(data.bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ  )
        m.addConstr(rev_mFRR_dw == qsum( (data.bid_mFRR_dw[t]) * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum( (data.bid_mFRR_up[t] )* pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # ---------------- Activation constraints ----------------
        m.addConstrs(p_bal[t] == data.bid_mFRR_dw[t] * act_dw[t] - data.bid_mFRR_up[t] * act_up[t] + slack_aux[t] for t in T)
        m.addConstrs(qsum(p_act[t,s] for s in S) == data.p_tot[t] + p_bal[t] for t in T)
        m.addConstrs(p_act[t, s] <= P_max[s] * z_act[t, s] for s in S for t in T)
        m.addConstrs(P_min[s] * z_act[t, s] <= p_act[t, s] for s in S for t in T)
        m.addConstrs(qsum(z_act[t, s] for s in S) <= 1 for t in T)

        m.addConstrs(p_tot_act[t] == qsum(p_act[t, s] for s in S) for t in T)

        m.addConstrs(z_on[t] == qsum(z_act[t, s] for s in S) for t in T)
        m.addConstrs(p_sb_act[t] == P_sb * z_sb[t] for t in T)
        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))
        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)

        # ---------------- Hydrogen constraints ----------------
        m.addConstrs(h_act[t] == qsum(A[s] *  p_act[t, s]  + B[s] * z_act[t, s] for s in S) * dT - slack_dw[t] + slack_up[t] for t in T)

        m.addConstr(H_min <= qsum(h_act[t] for t in T))

        m.addConstrs(h_act[t] == qsum(h_D[t, d] for d in D) for t in T)
        m.addConstrs(h_D[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)
        m.addConstrs(s_D[0, d] == h_D[0, d] for d in D)
        m.addConstrs(s_D[t, d] == h_D[t, d] + s_D[t - 1, d] for t in T1 for d in D)
        m.addConstrs(s_D[t, d] <= S_max for d in D for t in T)

        m.optimize()

        print('Obj: %g' % m.objVal)

        # Save in dataframe
        var_names_t = ['p_tot_act', 'p_sb_act', 'h_act', 'slack_dw', 'slack_up']

        real = pd.DataFrame()

        real['objective'] = pd.Series(m.objVal).repeat(NT).reset_index(drop=True) - penalty * sum(slack_up[i].x + slack_dw[i].x  for i in range(NT)) - + penalty * penalty * sum(slack_aux[t].x  for t in T)

        real['p_tot'] = data.p_tot
        real['p_sb'] = data.p_sb
        real['h_E'] = data.h_E

        for i in var_names_t:
            real[i] = [var.x for var in m.getVars() if i in var.VarName]

        p_acts = np.zeros((NT, NS))
        for s in S:
            column = 'p_act' + str(s)
            for t in T:
                p_acts[t, s] = p_act[t, s].x
            real[column] = p_acts[:, s]


        real = real.assign(act_up = act_up)
        real = real.assign(act_dw = act_dw)

        rev_var_names = ['rev_DA', 'rev_H2', 'rev_FCR', 'rev_mFRR_dw', 'rev_mFRR_up', 'rev_bal']

        revenues = pd.DataFrame(columns=rev_var_names)
        for i in rev_var_names:
            revenues[i] =  [var.x for var in m.getVars() if i in var.VarName]

        real['bid_DA'] = data.bid_DA
        real['pi_DA'] = pd.DataFrame(pi_DA)
        real['bid_mFRR_up'] = data.bid_mFRR_up
        real['pi_mFRR_up'] = pd.DataFrame(pi_mFRR_up)
        real['bid_mFRR_dw'] = data.bid_mFRR_dw
        real['pi_mFRR_dw'] = pd.DataFrame(pi_mFRR_dw)
        real['bid_FCR'] = data.bid_FCR
        real['pi_FCR'] = pd.DataFrame(pi_FCR)

        pwr_bal = np.zeros(NT)
        for t in T:
            pwr_bal[t] = p_bal[t].x
        real['pwr_bal'] = pd.DataFrame(pwr_bal)

        real['pi_bal'] = pd.DataFrame(pi_bal)


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return real, revenues



########################################################################################################################
# Run the model(s)
########################################################################################################################

# Crease structures to save results

results = {}        # results from DA optimization
ex_post = {}        # results from realization
revenues = {}       # realized revenues
hourly_results = {}
hourly_ex_post = {}
profit = {}         # realized profit

for j in range(len(model)):
    results[model[j]] = {}
    ex_post[model[j]] = {}
    revenues[model[j]] = {}

# Import time-series
pi_H2, df_full = import_full(case)

for n in range(Days):
    for j in range(len(model)):

        # Import time series for given day
        pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal,  act_dw, act_up = import_daily(df_full, n, 365)

        # Assume no down activation capacity is bought (see Energinet)
        act_dw = np.zeros(NT)

        # THE NO ANCILLARY SERVICE PROVISION CASE;  SET MAX AND MIN BIDS TO ZERO
        if model[j] == 'no_as':
            # make sure there are no reserve bids
            FCR_min = 0  # Minimum bid size in MW
            FCR_max = 0  # Maximum bid size in MW

            mFRR_up_min = 0  # Minimum bid size in MW
            mFRR_up_max = 0  # Maximum bid size in MW

            mFRR_dw_min = 0  # Minimum bid size in MW
            mFRR_dw_max = 0  # Maximum bid size in MW

            # Run optimization for bids
            data = AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw=np.repeat(0, NT), alpha_up=np.repeat(0, NT))
            results[model[j]]['Day_' + str(n)] = data

            # Run realization for bids
            ex_post[model[j]]['Day_' + str(n)], temp = realization_h2_slack(data, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal,
                                                               act_up, act_dw)
            revenues[model[j]]['Day_' + str(n)] = temp
            profit['Day_' + str(n), model[j]] = temp.loc[0]['rev_H2'] + temp.loc[0]['rev_FCR'] + temp.loc[0][
                'rev_mFRR_up'] + temp.loc[0]['rev_mFRR_dw'] - temp.loc[0]['rev_DA'] - temp.loc[0]['rev_bal']

        else:
            FCR_min = 2 * FCR_factor  # Minimum bid size in MW
            FCR_max = 10 * FCR_factor  # Maximum bid size in MW

            mFRR_up_min = 2 * mFRR_factor  # Minimum bid size in MW
            mFRR_up_max = 10 * mFRR_factor # Maximum bid size in MW

            mFRR_dw_min = 0  # Minimum bid size in MW
            mFRR_dw_max = 0  # Maximum bid size in MW

        # THE PERFECT FORESIGHT CASE
            if model[j] == 'alpha_r':
                # Run optimization for bids
                ex_post[model[j]]['Day_' + str(n)], temp = AS_ELY_perfect(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up = act_up, act_dw = act_dw)
                results[model[j]]['Day_' + str(n)] =  ex_post[model[j]]['Day_' + str(n)]
                revenues[model[j]]['Day_' + str(n)] = temp
                profit['Day_' + str(n), model[j]] = temp.loc[0]['rev_H2'] + temp.loc[0]['rev_FCR'] + temp.loc[0][
                    'rev_mFRR_up'] + temp.loc[0]['rev_mFRR_dw'] - temp.loc[0]['rev_DA'] - temp.loc[0]['rev_bal']
            else:

        # ROBUST CASE i.e. alpha = 1
                if model[j] == 'alpha_1':
                    # Run optimization for bids
                    data = AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw = np.repeat(1, NT), alpha_up = np.repeat(1, NT))
                    results[model[j]]['Day_' + str(n)] = data

        # UNAWARE CASE i.e. alpha = 0
                elif model[j] == 'alpha_0':
                    # Run optimization for bids
                    data = AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw = np.repeat(0, NT), alpha_up = np.repeat(0, NT))
                    results[model[j]]['Day_' + str(n)] = data

                # Run realization for bids (here with slack variable on the hydrogen demand - i.e. assuming the reserve must be provided
                ex_post[model[j]]['Day_' + str(n)], temp  = realization_h2_slack(data, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up = act_up, act_dw = act_dw)
                revenues[model[j]]['Day_' + str(n)] = temp
                profit['Day_' + str(n),model[j]] = temp.loc[0]['rev_H2'] + temp.loc[0]['rev_FCR'] + temp.loc[0]['rev_mFRR_up']  + temp.loc[0]['rev_mFRR_dw'] -  temp.loc[0]['rev_DA'] - temp.loc[0]['rev_bal']



########################################################################################################################
# Save the results in daily structures
########################################################################################################################

colm_keysList = [key for key in results]
days_keysList = [key for key in results[colm_keysList[0]]]

ex_post_keysList = [key for key in ex_post[colm_keysList[0]][days_keysList[0]]]
revenues_keysList = [key for key in revenues[colm_keysList[0]][days_keysList[0]]]

ex_post_daily = pd.DataFrame(index=days_keysList,columns=ex_post_keysList)
revenues_daily = pd.DataFrame(index=days_keysList,columns=revenues_keysList)

folder = dir + '/Results/' + case

for j in colm_keysList:
    for i in days_keysList:
        for k in ex_post_keysList:
            ex_post_daily[k][i] = sum(ex_post[j][i][k])
        ex_post_daily['objective'][i] = ex_post[j][i]['objective'][0]

        for k in revenues_keysList:
            revenues_daily[k][i] = revenues[j][i][k][0]

    if FCR_factor == 1 and mFRR_factor == 1:
        if not os.path.exists(folder + '/' + 'mFRR_FCR/' + j):
            os.makedirs(folder + '/' + 'mFRR_FCR/' + j)

        ex_post_daily.to_excel(folder + '/' + 'mFRR_FCR/' + j + '/' + j + '_Ex_post.xlsx')
        revenues_daily.to_excel(folder + '/' + 'mFRR_FCR/' + j + '/' + j + '_Revenues.xlsx')

    if FCR_factor == 1 and mFRR_factor == 0:
        if not os.path.exists(folder + '/' + 'FCR_only/' + j):
            os.makedirs(folder + '/' + 'FCR_only/' + j)

        ex_post_daily.to_excel(folder + '/' + 'FCR_only/' + j + '/' + j + '_Ex_post.xlsx')
        revenues_daily.to_excel(folder + '/' + 'FCR_only/' + j + '/' + j + '_Revenues.xlsx')

    if FCR_factor == 0 and mFRR_factor == 1:
        if not os.path.exists(folder + '/' + 'mFRR_only/' + j):
            os.makedirs(folder + '/' + 'mFRR_only/' + j)

        ex_post_daily.to_excel(folder + '/' + 'mFRR_only/' + j + '/' + j + '_Ex_post.xlsx')
        revenues_daily.to_excel(folder + '/' + 'mFRR_only/' + j + '/' + j + '_Revenues.xlsx')
