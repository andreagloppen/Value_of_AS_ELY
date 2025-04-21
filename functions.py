import numpy as np
import pandas as pd
import os
import time
import datetime as dt
import sys
sys.path.append(os.path.abspath("/Users/andrea/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/PhD/Code/PyCodes/AS_paper"))
from inputs import *

#########################################################
# Define time series import functions
#########################################################



# Import the time-series for the full case
def import_full(case):
    """ Import price and activation time series.
    """
    df_full = pd.DataFrame()
    if case == 'DK1_21':
        # Initial dataframe
        idx = pd.date_range('01-01-2021 00:00', '31-12-2021 23:00', freq='1h')
        df_full['Date_time'] = idx
        df_full['Test'] = 'T'
        df_full = df_full.set_index('Date_time')

        # DA prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/DA_DK1_2021.xlsx'),
                          columns=['HourDK', 'SpotPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "SpotPriceEUR": "pi_DA"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # FCR prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FcrDK1_2021.xlsx'),
                          columns=['HourDK', 'FCRdk_EUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "FCRdk_EUR": "pi_FCR"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR dw prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2021.xlsx'), columns=['HourDK', 'mFRR_DownPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_DownPriceEUR": "pi_mFRR_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR up prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2021.xlsx', decimal='.'),
                          columns=['HourDK', 'mFRR_UpPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_UpPriceEUR": "pi_mFRR_up"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # Imbalance prices, up and dw activation occurance
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2021.xlsx'),
                          columns=['HourDK', 'ImbalancePriceEUR', 'Act_up', 'Act_dw'])

        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(
            columns={"HourDK": "Date_time", "ImbalancePriceEUR": "pi_bal", "Act_up": "act_up", "Act_dw": "act_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # hydrogen price
        pi_H2 = 5

        # FCR activation signal
        df_FCR_act = pd.DataFrame(pd.read_csv('AS_paper/Data/Frequenz_DK1_2023.csv', sep=';', decimal=','))
        # df_FCR_act['DATE'] = pd.to_datetime(df_FCR_act['DATE'], format = '%d.%m.%Y')
        # df_FCR_act['TIME'] = pd.to_datetime(df_FCR_act['TIME'], format = '%H:%M:%S')

        df_FCR_act['DATE_TIME'] = pd.to_datetime(df_FCR_act['DATE'] + ' ' + df_FCR_act['TIME'],
                                                 format='%d.%m.%Y %H:%M:%S')

        df_FCR_act.set_index('DATE_TIME', inplace=True)
        df_FCR_act[' FREQUENCY_[HZ]'] = 50  # No activation signal found

    if case == 'DK1_22':
        # Initial dataframe
        idx = pd.date_range('01-01-2022 00:00', '31-12-2022 23:00', freq='1h')
        df_full['Date_time'] = idx
        df_full['Test'] = 'T'
        df_full = df_full.set_index('Date_time')

        # DA prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/DA_DK1_2022.xlsx'),
                          columns=['HourDK', 'SpotPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "SpotPriceEUR": "pi_DA"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # FCR prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FcrDK1_2022.xlsx'),
                          columns=['HourDK', 'FCRdk_EUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "FCRdk_EUR": "pi_FCR"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR dw prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2022.xlsx'), columns=['HourDK', 'mFRR_DownPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_DownPriceEUR": "pi_mFRR_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR up prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2022.xlsx', decimal='.'),
                          columns=['HourDK', 'mFRR_UpPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_UpPriceEUR": "pi_mFRR_up"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # Imbalance prices, up and dw activation occurance
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2022.xlsx'),
                          columns=['HourDK', 'ImbalancePriceEUR', 'Act_up', 'Act_dw'])

        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(
            columns={"HourDK": "Date_time", "ImbalancePriceEUR": "pi_bal", "Act_up": "act_up", "Act_dw": "act_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # hydrogen price
        pi_H2 = 5

        # FCR activation signal
        df_FCR_act = pd.DataFrame(pd.read_csv('AS_paper/Data/Frequenz_DK1_2023.csv', sep=';', decimal=','))
        # df_FCR_act['DATE'] = pd.to_datetime(df_FCR_act['DATE'], format = '%d.%m.%Y')
        # df_FCR_act['TIME'] = pd.to_datetime(df_FCR_act['TIME'], format = '%H:%M:%S')

        df_FCR_act['DATE_TIME'] = pd.to_datetime(df_FCR_act['DATE'] + ' ' + df_FCR_act['TIME'],
                                                 format='%d.%m.%Y %H:%M:%S')

        df_FCR_act.set_index('DATE_TIME', inplace=True)
        df_FCR_act[' FREQUENCY_[HZ]'] = 50 # No activation signal found



    if case == "DK1_23":

        # Initial dataframe
        idx = pd.date_range('01-01-2023 00:00', '31-12-2023 23:00', freq='1h')
        df_full['Date_time'] = idx
        df_full['Test'] = 'T'
        df_full = df_full.set_index('Date_time')

        # DA prices
        df = pd.DataFrame(pd.read_csv('AS_paper/Data/DA_DK1_2023.csv', sep=';', decimal='.'), columns=['HourDK', 'SpotPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'],format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "SpotPriceEUR": "pi_DA"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # FCR prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/FcrDK1_2023.xlsx'), columns=['HourDK', 'FCRdk_EUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "FCRdk_EUR": "pi_FCR"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR dw prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2023.xlsx'), columns=['HourDK', 'mFRR_DownPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_DownPriceEUR": "pi_mFRR_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # mFRR up prices
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/mFRR_DK1_2023.xlsx', decimal='.'), columns=['HourDK', 'mFRR_UpPriceEUR'])
        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "mFRR_UpPriceEUR": "pi_mFRR_up"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # Imbalance prices, up and dw activation occurance
        df = pd.DataFrame(pd.read_excel('AS_paper/Data/RegulatingBalancePowerdata_DK1_2023.xlsx'),
                          columns=['HourDK', 'ImbalancePriceEUR', 'Act_up', 'Act_dw'])

        df = df.drop_duplicates(subset=['HourDK'], keep='last')
        df['HourDK'] = pd.to_datetime(df['HourDK'], format='%Y-%m-%d %H:%M')
        df = df.rename(columns={"HourDK": "Date_time", "ImbalancePriceEUR": "pi_bal", "Act_up": "act_up", "Act_dw": "act_dw"})
        df = df.dropna(subset=['Date_time']).set_index('Date_time').asfreq('1h')

        df_full = pd.merge(df_full, df, how='left', left_index=True, right_index=True)

        # hydrogen price
        pi_H2 = 5


        # FCR activation signal
        df_FCR_act = pd.DataFrame(pd.read_csv('AS_paper/Data/Frequenz_DK1_2023.csv', sep=';', decimal=','))
        # df_FCR_act['DATE'] = pd.to_datetime(df_FCR_act['DATE'], format = '%d.%m.%Y')
        # df_FCR_act['TIME'] = pd.to_datetime(df_FCR_act['TIME'], format = '%H:%M:%S')

        df_FCR_act['DATE_TIME'] = pd.to_datetime(df_FCR_act['DATE'] + ' ' + df_FCR_act['TIME'],
                                                 format='%d.%m.%Y %H:%M:%S')

        df_FCR_act.set_index('DATE_TIME', inplace=True)

    # Specify that we only assume histroical activation if our price bid is low enough
    for i in idx:
        if df_full.loc[(i, 'pi_bal')] < 19 / pi_H2:
            df_full.loc[(i, 'act_dw')] = 0
            df_full.loc[(i, 'act_up')] = 0

    return pi_H2, df_full, df_FCR_act


# Import the time-series for the day d
def import_daily(df_full, df_FCR_act, d, Days):
    """ Outputs time-series for the current day d only.
    """
    pi_DA_mat = np.split(df_full['pi_DA'].astype(float).to_numpy(), 365)
    pi_FCR_mat = np.split(df_full['pi_FCR'].astype(float).to_numpy(), 365)
    pi_mFRR_dw_mat = np.split(df_full['pi_mFRR_dw'].astype(float).to_numpy(), 365)
    pi_mFRR_up_mat = np.split(df_full['pi_mFRR_up'].astype(float).to_numpy(), 365)
    pi_bal_mat = np.split(df_full['pi_bal'].astype(float).to_numpy(), 365)

    act_dw_mat = np.split(df_full['act_dw'].astype(float).to_numpy(), 365)
    act_up_mat = np.split(df_full['act_up'].astype(float).to_numpy(), 365)

    pi_DA = pi_DA_mat[d].flatten()
    pi_DA[np.isnan(pi_DA)] = 0

    pi_FCR = pi_FCR_mat[d].flatten()
    pi_FCR[np.isnan(pi_FCR)] = 0
    pi_FCR = pi_FCR[0::NJ]

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

    dates = df_FCR_act['DATE'].unique()

    df_FCR_act_date = df_FCR_act.loc[df_FCR_act['DATE'] == dates[d]]
    df_FCR_act_date.insert(3, 'Time_stamp',np.asarray(df_FCR_act_date.index))
    df_FCR_act_date = df_FCR_act_date.drop_duplicates(subset=['Time_stamp'], keep='last')

    df_FCR_act_date = df_FCR_act_date.dropna(subset=['Time_stamp']).set_index('Time_stamp').asfreq('1s')

    df_FCR_act_date = df_FCR_act_date.fillna(50.000)

    response_FCR = response_FCR_daily(df_FCR_act_date)

    return pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_dw, act_up, response_FCR


#########################################################
# Define optimization model in functions
#########################################################

def AS_ELY_all_inputs(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw, alpha_up,NJ,NI,J,I,mFRR_up_max,mFRR_up_min,mFRR_dw_max,mFRR_dw_min, FCR_max, FCR_min):
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("ELY_AS")

        p_E = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_E")  # electrolyzer power [MW]
        p_sb = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb")  # electrolyzer sb power [MW]
        p_tot = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="p_tot")  # electrolyzer power over segments [MW]

        p_act_dw = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_dw")  # dw act. power [MW]
        p_act_up = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_up")  # up act. power [MW]

        z_E = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_E")  # electrolyzer segment
        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act_dw = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_dw")  # electrolyzer segment
        z_act_up = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_up")  # electrolyzer segment

        # HYDROGEN VARIABLES
        h_E = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_E")  # expected hydrogen prod (no act) [kg]
        h_act_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_up")  # expected up-activated hydrogen prod [kg]
        h_act_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_dw")  # expected dw-activated hydrogen prod [kg]
        h_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="h_D_dw")  # expected dw-activated tube-trailer hydrogen input [kg]
        s_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="s_D_dw")  # expected dw-activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # Set objective
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       , GRB.MINIMIZE)  # (1)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_E[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ)
        m.addConstr(rev_mFRR_dw == qsum(bid_mFRR_dw[t] * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum(bid_mFRR_up[t] * pi_mFRR_up[t] for t in T) * dT)


        #m.addConstr(bid_mFRR_up[7] == 5)

        # ---------------- Electrolyzer constraints ----------------

        m.addConstrs(p_E[t, s] <= P_max[s] * z_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(P_min[s] * z_E[t, s] <= p_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(qsum(z_E[t, s] for s in S) <= 1 for t in T)  # (3)

        m.addConstrs(h_E[t] == qsum(A[s] * p_E[t, s] + B[s] * z_E[t, s] for s in S) * dT for t in T)  # (4)

        m.addConstrs(p_tot[t] == qsum(p_E[t, s] for s in S) for t in T)  # (5)

        m.addConstrs(z_on[t] == qsum(z_E[t, s] for s in S) for t in T)  # (6)

        m.addConstrs(p_sb[t] == z_sb[t] * P_sb for t in T)  # (7)

        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (8)

        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (9)

        # ---------------- Market-bids constraints ----------------

        # DA bid
        m.addConstrs(bid_DA[t] == dT * (p_tot[t] + p_sb[t]) for t in T)  # (10)

        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J)  # (11)
        m.addConstrs(
            bid_FCR[i] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J)  # (12)

        # mFRR down and up bids
        # TODO: Should FCR be limited to only occur when there is an on-state, while mFRR can occur both from on and sb?
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - p_tot[t] for t in T)  # (13)
        m.addConstrs(bid_mFRR_up[t] <= p_tot[t] - C_E * E_min * (1 - z_off[t]) for t in T)  # (14)

        # Sum of reserve bids
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j
            in J)  # (15)
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I
            for j in J)  # (16)

        # Bid size limitations
        m.addConstrs(FCR_min * z_FCR[i] <= bid_FCR[i] for i in I)  # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max * z_FCR[i] for i in I)  # (17)

        m.addConstrs(mFRR_dw_min * z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T)  # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max * z_mFRR_dw[t] for t in T)  # (18)

        m.addConstrs(mFRR_up_min * z_mFRR_up[t] <= bid_mFRR_up[t] for t in T)  # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max * z_mFRR_up[t] for t in T)  # (19)

        # Up-activation constraints
        m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - bid_mFRR_up[t] * alpha_up[t] for t in T)  # (20)
        #m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - (bid_mFRR_up[t] + bid_FCR * alpha_up[t] for t in T)  # (20)

        m.addConstrs(p_act_up[t, s] <= P_max[s] * z_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(P_min[s] * z_act_up[t, s] <= p_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(qsum(z_act_up[t, s] for s in S) == 1 for t in T)  # (22)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstr(H_min <= qsum(h_act_up[t] for t in T))  # (24)

        # Dw-activation constraints
        m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + bid_mFRR_dw[t] * alpha_dw[t] for t in T)  # (25)
        #m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + (bid_mFRR_dw[t] + bid_FCR[t] )* alpha_dw[t] for t in T)  # (25)

        m.addConstrs(p_act_dw[t, s] <= P_max[s] * z_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(P_min[s] * z_act_dw[t, s] <= p_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(qsum(z_act_dw[t, s] for s in S) == 1 for t in T)  # (27)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstrs(
            h_act_dw[t] == qsum(A[s] * p_act_dw[t, s] + B[s] * z_act_dw[t, s] for s in S) * dT for t in T)  # (28)

        m.addConstrs(h_act_dw[t] == qsum(h_D_dw[t, d] for d in D) for t in T)  # (29)
        m.addConstrs(h_D_dw[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  # (30)
        m.addConstrs(s_D_dw[0, d] == h_D_dw[0, d] for d in D)  # (31)
        m.addConstrs(s_D_dw[t, d] == h_D_dw[t, d] + s_D_dw[t - 1, d] for t in T1 for d in D)  # (32)
        m.addConstrs(s_D_dw[t, d] <= S_max for d in D for t in T)  # (33)

        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_tot', 'p_sb', 'h_E', 'z_on', 'z_sb', 'z_off', 'bid_mFRR_up', 'h_act_up', 'bid_mFRR_dw',
                       'h_act_dw', 'bid_DA']
        # , 'mFRR_up_C_rev', 'mFRR_dw_C_rev', 'mFRR_up_A_rev', 'mFRR_dw_A_rev']

        data = pd.DataFrame(0, index=np.arange(NT), columns=var_names_t)
        for i in var_names_t:
            data[i] = [var.x for var in m.getVars() if i in var.VarName]

        bid_FCR = [bid_FCR[i].x for i in I]
        data = data.assign(bid_FCR=np.repeat(bid_FCR, NJ))

        rev_FCR = rev_FCR.x
        data = data.assign(rev_FCR=np.repeat(rev_FCR, NT))

        """
        z_el = np.zeros((NT, NS))
        for s in S:
            column = 'z_el' + str(s)
            for t in T:
                z_el[t, s] = z_E[t, s].x
            data[column] = z_el[:, s]

        z_up = np.zeros((NT, NS))
        for s in S:
            column = 'z_up' + str(s)
            for t in T:
                z_up[t, s] = z_act_up[t, s].x
            data[column] = z_up[:, s]

        z_dw = np.zeros((NT, NS))
        for s in S:
            column = 'z_dw' + str(s)
            for t in T:
                z_dw[t, s] = z_act_dw[t, s].x
            data[column] = z_dw[:, s]
        """
        data = data.assign(alpha_up=alpha_up)
        data = data.assign(alpha_dw=alpha_dw)

        data['objective'] = m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return data

# Model formulation to place DA and reserve bids based on 'forecasts'
def AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw, alpha_up):
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("ELY_AS")

        p_E = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_E")  # electrolyzer power [MW]
        p_sb = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb")  # electrolyzer sb power [MW]
        p_tot = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="p_tot")  # electrolyzer power over segments [MW]

        p_act_dw = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_dw")  # dw act. power [MW]
        p_act_up = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_up")  # up act. power [MW]

        z_E = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_E")  # electrolyzer segment
        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act_dw = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_dw")  # electrolyzer segment
        z_act_up = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_up")  # electrolyzer segment

        # HYDROGEN VARIABLES
        h_E = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_E")  # expected hydrogen prod (no act) [kg]
        h_act_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_up")  # expected up-activated hydrogen prod [kg]
        h_act_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_dw")  # expected dw-activated hydrogen prod [kg]
        h_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="h_D_dw")  # expected dw-activated tube-trailer hydrogen input [kg]
        s_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="s_D_dw")  # expected dw-activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # Set objective
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       , GRB.MINIMIZE)  # (1)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_E[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ)
        m.addConstr(rev_mFRR_dw == qsum(bid_mFRR_dw[t] * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum(bid_mFRR_up[t] * pi_mFRR_up[t] for t in T) * dT)


        #m.addConstr(bid_mFRR_up[7] == 5)

        # ---------------- Electrolyzer constraints ----------------

        m.addConstrs(p_E[t, s] <= P_max[s] * z_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(P_min[s] * z_E[t, s] <= p_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(qsum(z_E[t, s] for s in S) <= 1 for t in T)  # (3)

        m.addConstrs(h_E[t] == qsum(A[s] * p_E[t, s] + B[s] * z_E[t, s] for s in S) * dT for t in T)  # (4)

        m.addConstrs(p_tot[t] == qsum(p_E[t, s] for s in S) for t in T)  # (5)

        m.addConstrs(z_on[t] == qsum(z_E[t, s] for s in S) for t in T)  # (6)

        m.addConstrs(p_sb[t] == z_sb[t] * P_sb for t in T)  # (7)

        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (8)

        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (9)

        # ---------------- Market-bids constraints ----------------

        # DA bid
        m.addConstrs(bid_DA[t] == dT * (p_tot[t] + p_sb[t]) for t in T)  # (10)

        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J)  # (11)
        m.addConstrs(
            bid_FCR[i] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J)  # (12)

        # mFRR down and up bids
        # TODO: Should FCR be limited to only occur when there is an on-state, while mFRR can occur both from on and sb?
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - p_tot[t] for t in T)  # (13)
        m.addConstrs(bid_mFRR_up[t] <= p_tot[t] - C_E * E_min * (1 - z_off[t]) for t in T)  # (14)

        # Sum of reserve bids
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j
            in J)  # (15)
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I
            for j in J)  # (16)

        # Bid size limitations
        m.addConstrs(FCR_min * z_FCR[i] <= bid_FCR[i] for i in I)  # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max * z_FCR[i] for i in I)  # (17)

        m.addConstrs(mFRR_dw_min * z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T)  # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max * z_mFRR_dw[t] for t in T)  # (18)

        m.addConstrs(mFRR_up_min * z_mFRR_up[t] <= bid_mFRR_up[t] for t in T)  # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max * z_mFRR_up[t] for t in T)  # (19)

        # Up-activation constraints
        m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - bid_mFRR_up[t] * alpha_up[t] for t in T)  # (20)
        #m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - (bid_mFRR_up[t] + bid_FCR * alpha_up[t] for t in T)  # (20)

        m.addConstrs(p_act_up[t, s] <= P_max[s] * z_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(P_min[s] * z_act_up[t, s] <= p_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(qsum(z_act_up[t, s] for s in S) == 1 for t in T)  # (22)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstr(H_min <= qsum(h_act_up[t] for t in T))  # (24)

        # Dw-activation constraints
        m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + bid_mFRR_dw[t] * alpha_dw[t] for t in T)  # (25)
        #m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + (bid_mFRR_dw[t] + bid_FCR[t] )* alpha_dw[t] for t in T)  # (25)

        m.addConstrs(p_act_dw[t, s] <= P_max[s] * z_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(P_min[s] * z_act_dw[t, s] <= p_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(qsum(z_act_dw[t, s] for s in S) == 1 for t in T)  # (27)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstrs(
            h_act_dw[t] == qsum(A[s] * p_act_dw[t, s] + B[s] * z_act_dw[t, s] for s in S) * dT for t in T)  # (28)

        m.addConstrs(h_act_dw[t] == qsum(h_D_dw[t, d] for d in D) for t in T)  # (29)
        m.addConstrs(h_D_dw[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  # (30)
        m.addConstrs(s_D_dw[0, d] == h_D_dw[0, d] for d in D)  # (31)
        m.addConstrs(s_D_dw[t, d] == h_D_dw[t, d] + s_D_dw[t - 1, d] for t in T1 for d in D)  # (32)
        m.addConstrs(s_D_dw[t, d] <= S_max for d in D for t in T)  # (33)

        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_tot', 'p_sb', 'h_E', 'z_on', 'z_sb', 'z_off', 'bid_mFRR_up', 'h_act_up', 'bid_mFRR_dw',
                       'h_act_dw', 'bid_DA']
        # , 'mFRR_up_C_rev', 'mFRR_dw_C_rev', 'mFRR_up_A_rev', 'mFRR_dw_A_rev']

        data = pd.DataFrame(0, index=np.arange(NT), columns=var_names_t)
        for i in var_names_t:
            data[i] = [var.x for var in m.getVars() if i in var.VarName]

        bid_FCR = [bid_FCR[i].x for i in I]
        data = data.assign(bid_FCR=np.repeat(bid_FCR, NJ))

        rev_FCR = rev_FCR.x
        data = data.assign(rev_FCR=np.repeat(rev_FCR, NT))

        """
        z_el = np.zeros((NT, NS))
        for s in S:
            column = 'z_el' + str(s)
            for t in T:
                z_el[t, s] = z_E[t, s].x
            data[column] = z_el[:, s]

        z_up = np.zeros((NT, NS))
        for s in S:
            column = 'z_up' + str(s)
            for t in T:
                z_up[t, s] = z_act_up[t, s].x
            data[column] = z_up[:, s]

        z_dw = np.zeros((NT, NS))
        for s in S:
            column = 'z_dw' + str(s)
            for t in T:
                z_dw[t, s] = z_act_dw[t, s].x
            data[column] = z_dw[:, s]
        """
        data = data.assign(alpha_up=alpha_up)
        data = data.assign(alpha_dw=alpha_dw)

        data['objective'] = m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return data

def AS_ELY_perfect3(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw, alpha_up):
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("ELY_AS_PERFECT")

        p_E = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_E")  # electrolyzer power [MW]
        p_sb = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb")  # electrolyzer sb power [MW]
        p_tot = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="p_tot")  # electrolyzer power over segments [MW]

        p_act_dw = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_dw")  # dw act. power [MW]
        p_act_up = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act_up")  # up act. power [MW]

        z_E = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_E")  # electrolyzer segment
        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act_dw = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_dw")  # electrolyzer segment
        z_act_up = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act_up")  # electrolyzer segment

        # HYDROGEN VARIABLES
        h_E = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_E")  # expected hydrogen prod (no act) [kg]
        h_act_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_up")  # expected up-activated hydrogen prod [kg]
        h_act_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="h_act_dw")  # expected dw-activated hydrogen prod [kg]
        h_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="h_D_dw")  # expected dw-activated tube-trailer hydrogen input [kg]
        s_D_dw = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="s_D_dw")  # expected dw-activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # Set objective
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       - rev_bal  # Revenue from balancing upon activation - note that the price is given per hour
                       , GRB.MINIMIZE)  # (1)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act_up[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ)
        m.addConstr(rev_mFRR_dw == qsum(bid_mFRR_dw[t] * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum(bid_mFRR_up[t] * pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum((bid_mFRR_up[t] *alpha_up[t] - bid_mFRR_dw[t] *alpha_dw[t] )* pi_bal[t] for t in T) * dT)


        # ---------------- Electrolyzer constraints ----------------

        m.addConstrs(p_E[t, s] <= P_max[s] * z_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(P_min[s] * z_E[t, s] <= p_E[t, s] for s in S for t in T)  # (2)
        m.addConstrs(qsum(z_E[t, s] for s in S) <= 1 for t in T)  # (3)

        m.addConstrs(h_E[t] == qsum(A[s] * p_E[t, s] + B[s] * z_E[t, s] for s in S) * dT for t in T)  # (4)

        m.addConstrs(p_tot[t] == qsum(p_E[t, s] for s in S) for t in T)  # (5)

        m.addConstrs(z_on[t] == qsum(z_E[t, s] for s in S) for t in T)  # (6)

        m.addConstrs(p_sb[t] == z_sb[t] * P_sb for t in T)  # (7)

        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (8)

        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (9)

        # ---------------- Market-bids constraints ----------------

        # DA bid
        m.addConstrs(bid_DA[t] == dT * (p_tot[t] + p_sb[t]) for t in T)  # (10)

        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J)  # (11)
        m.addConstrs(
            bid_FCR[i] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J)  # (12)

        # mFRR down and up bids
        # TODO: Should FCR be limited to only occur when there is an on-state, while mFRR can occur both from on and sb?
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - p_tot[t] for t in T)  # (13)
        m.addConstrs(bid_mFRR_up[t] <= p_tot[t] - C_E * E_min * (1 - z_off[t]) for t in T)  # (14)

        # Sum of reserve bids
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j
            in J)  # (15)
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I
            for j in J)  # (16)

        # Bid size limitations
        m.addConstrs(FCR_min * z_FCR[i] <= bid_FCR[i] for i in I)  # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max * z_FCR[i] for i in I)  # (17)

        m.addConstrs(mFRR_dw_min * z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T)  # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max * z_mFRR_dw[t] for t in T)  # (18)

        m.addConstrs(mFRR_up_min * z_mFRR_up[t] <= bid_mFRR_up[t] for t in T)  # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max * z_mFRR_up[t] for t in T)  # (19)

        # Up-activation constraints
        m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - bid_mFRR_up[t] * alpha_up[t] for t in T)  # (20)
        #m.addConstrs(qsum(p_act_up[t, s] for s in S) == p_tot[t] - (bid_mFRR_up[t] + bid_FCR * alpha_up[t] for t in T)  # (20)

        m.addConstrs(p_act_up[t, s] <= P_max[s] * z_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(P_min[s] * z_act_up[t, s] <= p_act_up[t, s] for s in S for t in T)  # (21)
        m.addConstrs(qsum(z_act_up[t, s] for s in S) == 1 for t in T)  # (22)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstr(H_min <= qsum(h_act_up[t] for t in T))  # (24)

        # Dw-activation constraints
        m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + bid_mFRR_dw[t] * alpha_dw[t] for t in T)  # (25)
        #m.addConstrs(qsum(p_act_dw[t, s] for s in S) == p_tot[t] + (bid_mFRR_dw[t] + bid_FCR[t] )* alpha_dw[t] for t in T)  # (25)

        m.addConstrs(p_act_dw[t, s] <= P_max[s] * z_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(P_min[s] * z_act_dw[t, s] <= p_act_dw[t, s] for s in S for t in T)  # (26)
        m.addConstrs(qsum(z_act_dw[t, s] for s in S) == 1 for t in T)  # (27)

        m.addConstrs(
            h_act_up[t] == qsum(A[s] * p_act_up[t, s] + B[s] * z_act_up[t, s] for s in S) * dT for t in T)  # (23)

        m.addConstrs(
            h_act_dw[t] == qsum(A[s] * p_act_dw[t, s] + B[s] * z_act_dw[t, s] for s in S) * dT for t in T)  # (28)

        m.addConstrs(h_act_dw[t] == qsum(h_D_dw[t, d] for d in D) for t in T)  # (29)
        m.addConstrs(h_D_dw[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  # (30)
        m.addConstrs(s_D_dw[0, d] == h_D_dw[0, d] for d in D)  # (31)
        m.addConstrs(s_D_dw[t, d] == h_D_dw[t, d] + s_D_dw[t - 1, d] for t in T1 for d in D)  # (32)
        m.addConstrs(s_D_dw[t, d] <= S_max for d in D for t in T)  # (33)

        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_tot', 'p_sb', 'h_E', 'z_on', 'z_sb', 'z_off', 'bid_mFRR_up', 'h_act_up', 'bid_mFRR_dw',
                       'h_act_dw', 'bid_DA']
        # , 'mFRR_up_C_rev', 'mFRR_dw_C_rev', 'mFRR_up_A_rev', 'mFRR_dw_A_rev']

        data = pd.DataFrame(0, index=np.arange(NT), columns=var_names_t)
        for i in var_names_t:
            data[i] = [var.x for var in m.getVars() if i in var.VarName]

        bid_FCR = [bid_FCR[i].x for i in I]
        data = data.assign(bid_FCR=np.repeat(bid_FCR, NJ))
        """
        z_el = np.zeros((NT, NS))
        for s in S:
            column = 'z_el' + str(s)
            for t in T:
                z_el[t, s] = z_E[t, s].x
            data[column] = z_el[:, s]

        z_up = np.zeros((NT, NS))
        for s in S:
            column = 'z_up' + str(s)
            for t in T:
                z_up[t, s] = z_act_up[t, s].x
            data[column] = z_up[:, s]

        z_dw = np.zeros((NT, NS))
        for s in S:
            column = 'z_dw' + str(s)
            for t in T:
                z_dw[t, s] = z_act_dw[t, s].x
            data[column] = z_dw[:, s]
        """
        data = data.assign(alpha_up=alpha_up)
        data = data.assign(alpha_dw=alpha_dw)

        data['objective'] = m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return data


# Perfect foresight on activation and balancing
def AS_ELY_perfect(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_dw, act_up):
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("ELY_AS")

        p_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_DA")  # electrolyzer DA power [MW]
        p_sb = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_sb")  # electrolyzer sb power [MW]
        p_tot = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="p_tot")  # electrolyzer power over segments [MW]
        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_bal = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # bal. power [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer segment

        # HYDROGEN VARIABLES
        h_E = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_E")  # expected activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_D")  # expected activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="s_D")  # expected activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        # rev_mFRR_dw_act = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_dw_act")  #
        # rev_mFRR_up_act = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_mFRR_up_act")  #
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="rev_bal")  # bal [DKK]
        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # Set objective
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal  # Cost of down activation [DKK]
                       , GRB.MINIMIZE)  # (1)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_E[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ)
        m.addConstr(rev_mFRR_dw == qsum(bid_mFRR_dw[t] * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum(bid_mFRR_up[t] * pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # m.addConstr(rev_mFRR_dw_act == qsum(bid_mFRR_dw[t] * pi_bal[t] * act_dw[t] for t in T))
        # m.addConstr(rev_mFRR_up_act == qsum(bid_mFRR_up[t] * pi_bal[t] * act_up[t] for t in T))

        # ---------------- Market-bids constraints ----------------

        # DA bid
        m.addConstrs(bid_DA[t] == dT * (p_DA[t]) for t in T)  # (10)

        # FCR bid
        m.addConstrs(bid_FCR[i] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j in J)  # (11)
        m.addConstrs(
            bid_FCR[i] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I for j in J)  # (12)

        # mFRR down and up bids
        # TODO: Should FCR be limited to only occur when there is an on-state, while mFRR can occur both from on and sb?
        m.addConstrs(bid_mFRR_dw[t] <= C_E * (1 - z_off[t]) - p_tot[t] for t in T)  # (13)
        m.addConstrs(bid_mFRR_up[t] <= p_tot[t] - C_E * E_min * (1 - z_off[t]) for t in T)  # (14)

        # Sum of reserve bids
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_dw[i * NJ + j] <= C_E * (1 - z_off[i * NJ + j]) - p_tot[i * NJ + j] for i in I for j
            in J)  # (15)
        m.addConstrs(
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= p_tot[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in I
            for j in J)  # (16)

        # Bid size limitations
        m.addConstrs(FCR_min * z_FCR[i] <= bid_FCR[i] for i in I)  # (17)
        m.addConstrs(bid_FCR[i] <= FCR_max * z_FCR[i] for i in I)  # (17)

        m.addConstrs(mFRR_dw_min * z_mFRR_dw[t] <= bid_mFRR_dw[t] for t in T)  # (18)
        m.addConstrs(bid_mFRR_dw[t] <= mFRR_dw_max * z_mFRR_dw[t] for t in T)  # (18)

        m.addConstrs(mFRR_up_min * z_mFRR_up[t] <= bid_mFRR_up[t] for t in T)  # (19)
        m.addConstrs(bid_mFRR_up[t] <= mFRR_up_max * z_mFRR_up[t] for t in T)  # (19)

        m.addConstrs(p_bal[t] == bid_mFRR_dw[t] * act_dw[t] - bid_mFRR_up[t] * act_up[t] for t in T)
        m.addConstrs(qsum(p_act[t, s] for s in S) == p_DA[t] + p_bal[t] for t in T)

        m.addConstrs(p_act[t, s] <= P_max[s] * z_act[t, s] for s in S for t in T)  #
        m.addConstrs(P_min[s] * z_act[t, s] <= p_act[t, s] for s in S for t in T)  #
        m.addConstrs(qsum(z_act[t, s] for s in S) <= 1 for t in T)  # (3)

        m.addConstrs(p_tot[t] == qsum(p_act[t, s] for s in S) for t in T)  # (2)

        m.addConstrs(z_on[t] == qsum(z_act[t, s] for s in S) for t in T)  # (5)
        m.addConstrs(p_sb[t] == P_sb * z_sb[t] for t in T)  # (6)
        m.addConstrs(z_off[t] - z_off[t - 1] <= z_off[tau]
                     for t in range(1, NT - N_down - 1) for tau in range(t + 1, t + N_down))  # (7)
        m.addConstrs(z_on[t] + z_sb[t] + z_off[t] == 1 for t in T)  # (8)

        # ---------------- Hydrogen constraints ----------------
        m.addConstrs(h_E[t] == qsum(A[s] * p_act[t, s] + B[s] * z_act[t, s] for s in S) * dT for t in T)

        m.addConstr(H_min <= qsum(h_E[t] for t in T))

        m.addConstrs(h_E[t] == qsum(h_D[t, d] for d in D) for t in T)  #
        m.addConstrs(h_D[t, d] <= C_D * TT_bin[t, d] for t in T for d in D)  #
        m.addConstrs(s_D[0, d] == h_D[0, d] for d in D)  #
        m.addConstrs(s_D[t, d] == h_D[t, d] + s_D[t - 1, d] for t in T1 for d in D)  #
        m.addConstrs(s_D[t, d] <= S_max for d in D for t in T)
        m.optimize()

        print('Obj: %g' % m.objVal)

        var_names_t = ['p_DA', 'p_tot', 'p_sb', 'z_on', 'z_sb', 'z_off', 'bid_mFRR_up', 'bid_mFRR_dw', 'h_E', 'bid_DA',
                       'p_bal']
        # , 'mFRR_up_C_rev', 'mFRR_dw_C_rev', 'mFRR_up_A_rev', 'mFRR_dw_A_rev']

        data = pd.DataFrame(0, index=np.arange(NT), columns=var_names_t)
        for i in var_names_t:
            data[i] = [var.x for var in m.getVars() if i in var.VarName]

        bid_FCR = [bid_FCR[i].x for i in I]
        data = data.assign(bid_FCR=np.repeat(bid_FCR, NJ))

        z_E_act = np.zeros((NT, NS))
        for s in S:
            column = 'z_act' + str(s)
            for t in T:
                z_E_act[t, s] = z_act[t, s].x
            data[column] = z_E_act[:, s]

        z_el = np.zeros((NT, NS))
        for s in S:
            column = 'z_el' + str(s)
            for t in T:
                z_el[t, s] = z_act[t, s].x
            data[column] = z_el[:, s]

        z_up = np.zeros((NT, NS))
        for s in S:
            column = 'z_up' + str(s)
            for t in T:
                z_up[t, s] = z_act[t, s].x
            data[column] = z_up[:, s]

        z_dw = np.zeros((NT, NS))
        for s in S:
            column = 'z_dw' + str(s)
            for t in T:
                z_dw[t, s] = z_act[t, s].x
            data[column] = z_dw[:, s]

        data = data.assign(h_act_up=h_E)
        data = data.assign(h_act_dw=h_E)

        data = data.assign(alpha_up=act_up)
        data = data.assign(alpha_dw=act_dw)

        data['objective'] = m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    print("--- %s seconds ---" % (time.time() - start_time))

    return data


# Realization of activation
def AS_ELY_perfect2(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up, act_dw):
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("REALIZATION")

        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_tot_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot_act")  # act. power [MW]
        p_sb_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="p_sb_act")  # electrolyzer sb power [MW]

        p_bal = m.addVars(NT, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # imbalance [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer activated segment

        slack_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="slack_dw")  # slack on activation [MW]
        slack_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="slack_up")  # slack on activation [MW]

        # HYDROGEN VARIABLES
        h_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act")  # activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_D")  # activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="s_D")  # activated tube-trailer storage state [kg]

        # MARKET BIDS
        bid_DA = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_DA")  # Day-ahead energy bid [kWh]
        bid_FCR = m.addVars(NI, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_FCR")  # FCR bid [kW]
        bid_mFRR_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_up")  # mFRR up bid [kW]
        bid_mFRR_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="bid_mFRR_dw")  # mFRR dw bid [kw]

        z_FCR = m.addVars(NI, vtype=GRB.BINARY, name="z_FCR")  # FCR bid state
        z_mFRR_dw = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_dw")  # mFRR_dw bid state
        z_mFRR_up = m.addVars(NT, vtype=GRB.BINARY, name="z_mFRR_up")  # mFRR_up bid state

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_bal")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # ---------------- Set Objective ----------------
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw * dT  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up * dT  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal  # Revenue from delta energy (can be both positive [dw-reg] and negative [up-reg])
                       + 100000 * qsum(slack_up[t] + slack_dw[t] for t in T)
                       , GRB.MINIMIZE)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NI)
        m.addConstr(rev_mFRR_dw == qsum((bid_mFRR_dw[t] - slack_dw[t]) * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum((bid_mFRR_up[t] - slack_up[t]) * pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # ---------------- Activation constraints ----------------
        m.addConstrs(
            p_bal[t] == bid_mFRR_dw[t] * act_dw[t] - slack_dw[t] - bid_mFRR_up[t] * act_up[t] + slack_up[t] for t in T)
        m.addConstrs(qsum(p_act[t, s] for s in S) == bid_DA[t] + p_bal[t] for t in T)
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
            bid_FCR[i] + bid_mFRR_up[i * NJ + j] <= bid_DA[i * NJ + j] - C_E * E_min * (1 - z_off[i * NJ + j]) for i in
            I
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

        real['objective'] = pd.Series(m.objVal).repeat(NT).reset_index(drop=True) - 100000 * sum(
            slack_up[i].x + slack_dw[i].x for i in range(NT))

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

        real = real.assign(act_up=act_up)
        real = real.assign(act_dw=act_dw)

        rev_var_names = ['rev_DA', 'rev_H2', 'rev_FCR', 'rev_mFRR_dw', 'rev_mFRR_up', 'rev_bal']

        revenues = pd.DataFrame(columns=rev_var_names)
        for i in rev_var_names:
            revenues[i] = [var.x for var in m.getVars() if i in var.VarName]

        real['rev_H2_real'] = pd.DataFrame(h_act[t].x for t in T)
        real['bid_DA'] = pd.DataFrame(bid_DA[t].x for t in T)
        real['pi_DA'] = pd.DataFrame(pi_DA)
        real['bid_mFRR_up'] = pd.DataFrame(bid_mFRR_up[t].x for t in T)
        real['pi_mFRR_up'] = pd.DataFrame(pi_mFRR_up)
        real['bid_mFRR_dw'] = pd.DataFrame(bid_mFRR_dw[t].x for t in T)
        real['pi_mFRR_dw'] = pd.DataFrame(pi_mFRR_dw)
        real['bid_FCR'] = pd.Series(bid_FCR[i].x for i in I).repeat(NI).reset_index(drop=True)
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
def realization_h2_slack(data, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_up, act_dw, response):
    bid_FCR = data.bid_FCR[::NJ].reset_index(drop=True)
    penalty = 100000

    print('Day ' + str(d))

    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import quicksum as qsum
    start_time = time.time()

    try:

        # Create a new model
        m = gp.Model("REALIZATION")
        m.Params.LogToConsole = 0

        p_act = m.addVars(NT, NS, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_act")  # act. power [MW]
        p_tot_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_tot_act")  # act. power [MW]
        p_sb_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="p_sb_act")  # electrolyzer sb power [MW]

        p_bal = m.addVars(NT, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="p_bal")  # imbalance [MW]

        z_on = m.addVars(NT, vtype=GRB.BINARY, name="z_on")  # electrolyzer on state
        z_sb = m.addVars(NT, vtype=GRB.BINARY, name="z_sb")  # electorlyzer standby state
        z_off = m.addVars(NT, vtype=GRB.BINARY, name="z_off")  # electrolyzer off state

        z_act = m.addVars(NT, NS, vtype=GRB.BINARY, name="z_act")  # electrolyzer activated segment

        slack_dw = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="slack_dw")  # slack on activation [MW]
        slack_up = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                             name="slack_up")  # slack on activation [MW]

        slack_aux = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                              name="slack_aux")  # slack on activation [MW]

        # HYDROGEN VARIABLES
        h_act = m.addVars(NT, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name="h_act")  # activated hydrogen prod [kg]
        h_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="h_D")  # activated tube-trailer hydrogen input [kg]
        s_D = m.addVars(NT, ND, lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                        name="s_D")  # activated tube-trailer storage state [kg]

        # REVENUE STREAMS
        rev_DA = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_DA")  # Day-ahead revenue [DKK]
        rev_H2 = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                          name="rev_H2")  # Hydrogen revenue [DKK]
        rev_FCR = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_FCR")  # FRC reserve revenue [DKK]
        rev_mFRR_dw = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_dw")  # mFRR dw reserve revenue [DKK]
        rev_mFRR_up = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                               name="rev_mFRR_up")  # mFRR up reserve revenue [DKK]
        rev_bal = m.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS,
                           name="rev_bal")  # mFRR up reserve revenue [DKK]

        # MIPGAP

        m.setParam("MIPGap", mipgap)
        m.setParam('TimeLimit', timelimit)

        # ---------------- Set Objective ----------------
        m.setObjective(rev_DA  # Cost of DA bid [DKK]
                       - rev_H2  # Revenue hydrogen prod [DKK]
                       - rev_FCR  # Revenue from FCR bid [DKK] - note that the price is per hour (aka more rev)
                       - rev_mFRR_dw * dT  # Revenue from mFRR dw bid [DKK] - note that the price is given per hour
                       - rev_mFRR_up * dT  # Revenue from mFRR up bid [DKK] - note that the price is given per hour
                       + rev_bal  # Revenue from delta energy (can be both positive [dw-reg] and negative [up-reg])
                       + penalty * qsum(slack_up[t] + slack_dw[t] for t in T)
                       + penalty * penalty * qsum(slack_aux[t] for t in T)
                       , GRB.MINIMIZE)

        # ---------------- Revenue auxiliary variables ----------------
        m.addConstr(rev_DA == qsum(data.bid_DA[t] * pi_DA[t] for t in T))
        m.addConstr(rev_H2 == qsum(pi_H2 * h_act[t] for t in T))
        m.addConstr(rev_FCR == qsum(bid_FCR[i] * pi_FCR[i] for i in I) * NJ)
        m.addConstr(rev_mFRR_dw == qsum((data.bid_mFRR_dw[t]) * pi_mFRR_dw[t] for t in T) * dT)
        m.addConstr(rev_mFRR_up == qsum((data.bid_mFRR_up[t]) * pi_mFRR_up[t] for t in T) * dT)
        m.addConstr(rev_bal == qsum(p_bal[t] * pi_bal[t] for t in T) * dT)

        # ---------------- Activation constraints ----------------
        m.addConstrs(
            p_bal[t] == data.bid_mFRR_dw[t] * act_dw[t] - data.bid_mFRR_up[t] * act_up[t] + slack_aux[t] for t in T)
        m.addConstrs(qsum(p_act[t, s] for s in S) == data.p_tot[t] + p_bal[t] for t in T)
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
        m.addConstrs(
            h_act[t] == qsum(A[s] * p_act[t, s] + B[s] * z_act[t, s] for s in S) * dT - slack_dw[t] + slack_up[t] for t
            in T)

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

        real['objective'] = pd.Series(m.objVal).repeat(NT).reset_index(drop=True) - penalty * sum(
            slack_up[i].x + slack_dw[i].x for i in range(NT)) - + penalty * penalty * sum(slack_aux[t].x for t in T)

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

        real = real.assign(act_up=act_up)
        real = real.assign(act_dw=act_dw)

        rev_var_names = ['rev_DA', 'rev_H2', 'rev_FCR', 'rev_mFRR_dw', 'rev_mFRR_up', 'rev_bal']

        revenues = pd.DataFrame(columns=rev_var_names)
        for i in rev_var_names:
            revenues[i] = [var.x for var in m.getVars() if i in var.VarName]

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

    newdf = pd.DataFrame(np.repeat(real.values, 20, axis=0))
    newdf.columns = real.columns

    if len(response) < len(newdf["bid_FCR"]):
        n = len(newdf["bid_FCR"]) - len(response)
        response = np.append(response, response[-n:])

    elif len(response) > len(newdf["bid_FCR"]):
        n = len(response) - len(newdf["bid_FCR"])
        response = response[:-n]

    fcr_h2_response = newdf["bid_FCR"] * response * 19 * (1 / 20)
    fcr_h2_response_h = fcr_h2_response.groupby(fcr_h2_response.index // 20).sum()

    real['fcr_h2_response'] = fcr_h2_response_h

    return real, revenues


def realization_h2_version_2(data, act_up, act_dw, response, pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal):
    p_planned = np.repeat(data['p_tot'], (60 * 60))

    mFRR_dw_response = np.repeat(data['bid_mFRR_dw'] * act_dw, (60 * 60))
    mFRR_up_response = np.repeat(data['bid_mFRR_up'] * act_up, (60 * 60))
    FCR_response = np.repeat(data['bid_FCR'], (60 * 60)) * response

    p_real = p_planned + mFRR_dw_response - mFRR_up_response + FCR_response

    h_real = [min(A * i + B) / (60 * 60) for i in np.asarray(p_real)]

    p_real_hourly = pd.Series(p_real).groupby(p_real.index).sum() / (60 * 60)
    h_real_hourly = pd.Series(h_real).groupby(p_real.index).sum()
    p_mFRR_dw_hourly = pd.Series(mFRR_dw_response).groupby(p_real.index).sum() / (60 * 60)
    p_mFRR_up_hourly = pd.Series(mFRR_dw_response).groupby(p_real.index).sum() / (60 * 60)
    p_FCR_hourly = pd.Series(FCR_response).groupby(p_real.index).sum() / (60 * 60)

    real = pd.DataFrame()

    real['p_E'] = data['p_tot']
    real['p_real'] = p_real_hourly
    real['h_E'] = data['h_E']
    real['h_real'] = h_real_hourly

    real['p_FCR'] = p_FCR_hourly
    real['p_mFRR_dw'] = p_mFRR_dw_hourly
    real['p_mFRR_up'] = p_mFRR_up_hourly

    real['bid_DA'] = data.bid_DA
    real['pi_DA'] = pd.DataFrame(pi_DA)
    real['bid_mFRR_up'] = data.bid_mFRR_up
    real['pi_mFRR_up'] = pd.DataFrame(pi_mFRR_up)
    real['bid_mFRR_dw'] = data.bid_mFRR_dw
    real['pi_mFRR_dw'] = pd.DataFrame(pi_mFRR_dw)
    real['bid_FCR'] = data.bid_FCR
    real['pi_FCR'] = pd.DataFrame(np.repeat(pi_FCR, NJ))

    real['rev_H2_E'] = real['h_E'] * pi_H2
    real['rev_H2_real'] = real['h_real'] * pi_H2

    real['rev_DA'] = real['bid_DA'] * real['pi_DA']
    real['rev_FCR'] = real['bid_FCR'] * real['pi_FCR']
    real['rev_mFRR_dw'] = real['bid_mFRR_dw'] * real['pi_mFRR_dw']
    real['rev_mFRR_up'] = real['bid_mFRR_up'] * real['pi_mFRR_up']
    real['rev_bal'] = (real['p_E'] - real['p_real']) * pi_bal

    return real, real


# FCR activation response based on frequency
def response_FCR_daily(df_FCR_act_date):
    f_FCR_act_min = 49.90
    f_FCR_act_max = 50.10
    f_FCR_act_ntr = 50.00

    # freq = df_FCR_act_date["Frequency - real time data"]
    freq = df_FCR_act_date["FREQUENCY_[HZ]"].astype(float).values

    response = np.zeros(len(freq))

    for i in range(len(freq)):

        if freq[i] <= f_FCR_act_min:
            response[i] = - 1

        elif freq[i] >= f_FCR_act_max:
            response[i] = 1

        else:
            response[i] = (freq[i] - f_FCR_act_ntr) / (f_FCR_act_max - f_FCR_act_ntr)

    return response

