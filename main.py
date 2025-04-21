import numpy as np
import pandas as pd
import time
import datetime as dt
import sys
import os
sys.path.append(os.path.abspath("/Users/andrea/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/PhD/Code/PyCodes/AS_paper"))
from inputs import *
from functions import *

# UNITS: MW, MWh and kg (H2)

#########################################################
# Run the model(s)
#########################################################

# Crease structures to save results
for i in range(len(folders)):
    folder = folders[i]
    H_min = H_mins[i]
    results = {}  # results from DA optimization
    ex_post = {}  # results from realization
    revenues = {}  # realized revenues
    hourly_results = {}
    hourly_ex_post = {}
    profit = {}  # realized profit

    for j in range(len(model)):
        results[model[j]] = {}
        ex_post[model[j]] = {}
        revenues[model[j]] = {}

    # Import time-series
    pi_H2, df_full, df_FCR_act = import_full(case)

    for d in range(Days):
        for j in range(len(model)):

            # Import time series for given day
            pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, pi_bal, act_dw, act_up, response = import_daily(df_full, df_FCR_act, d,
                                                                                                   364)

            # Assume no down activation capacity is bought (see Energinet)
            act_dw = np.zeros(NT)

            # THE NO ANCILLARY SERVICE PROVISION CASE; JUST SET MAX AND MIN BIDS TO ZERO
            if model[j] == 'no_as':
                # make sure there are no reserve bids
                FCR_min = 0  # Minimum bid size in MW
                FCR_max = 0  # Maximum bid size in MW

                mFRR_up_min = 0  # Minimum bid size in MW
                mFRR_up_max = 0  # Maximum bid size in MW

                mFRR_dw_min = 0  # Minimum bid size in MW
                mFRR_dw_max = 0  # Maximum bid size in MW

                # Run optimization for bids
                data = AS_ELY_all_inputs(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw=np.repeat(0, NT), alpha_up=np.repeat(0, NT))
                results[model[j]]['Day_' + str(d)] = data

                # Run realization for bids
                ex_post[model[j]]['Day_' + str(d)], temp = realization_h2_version_2(data, act_up, act_dw, response,
                                                                                    pi_DA, pi_FCR, pi_mFRR_dw,
                                                                                    pi_mFRR_up, pi_bal)
                revenues[model[j]]['Day_' + str(d)] = temp

                profit['Day_' + str(d), model[j]] = temp.loc[0]['rev_H2_real'] + temp.loc[0]['rev_FCR'] + temp.loc[0][
                    'rev_mFRR_up'] + temp.loc[0]['rev_mFRR_dw'] - temp.loc[0]['rev_DA'] - temp.loc[0]['rev_bal']
            else:
                FCR_min = 2 * FCR_factor  # Minimum bid size in MW
                FCR_max = 10 * FCR_factor  # Maximum bid size in MW

                mFRR_up_min = 2 * mFRR_factor  # Minimum bid size in MW
                mFRR_up_max = 10 * mFRR_factor  # Maximum bid size in MW

                mFRR_dw_min = 0  # Minimum bid size in MW
                mFRR_dw_max = 0  # Maximum bid size in MW



                # ROBUST CASE
                if model[j] == 'alpha_1':
                    # Run optimization for bids
                    data = AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw=np.repeat(1, NT),
                                  alpha_up=np.repeat(1, NT))
                    results[model[j]]['Day_' + str(d)] = data

                # IGNORANT CASE
                elif model[j] == 'alpha_0':
                    # Run optimization for bids
                    data = AS_ELY(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw=np.repeat(0, NT),
                                  alpha_up=np.repeat(0, NT))
                    results[model[j]]['Day_' + str(d)] = data

                # PERFECT CASE
                elif model[j] == 'alpha_r':
                    # Run optimization for bids
                    data = AS_ELY_perfect3(pi_DA, pi_FCR, pi_mFRR_dw, pi_mFRR_up, alpha_dw=act_dw,
                                  alpha_up=act_up)
                    results[model[j]]['Day_' + str(d)] = data


                print('Day ' + str(d))
                ex_post[model[j]]['Day_' + str(d)], temp = realization_h2_version_2(data, act_up, act_dw, response,
                                                                                    pi_DA, pi_FCR, pi_mFRR_dw,
                                                                                    pi_mFRR_up, pi_bal)
                revenues[model[j]]['Day_' + str(d)] = temp

                profit['Day_' + str(d), model[j]] = temp.loc[0]['rev_H2_real'] + temp.loc[0]['rev_FCR'] + temp.loc[0][
                    'rev_mFRR_up'] + temp.loc[0]['rev_mFRR_dw'] - temp.loc[0]['rev_DA'] - temp.loc[0]['rev_bal']


    colm_keysList = [key for key in results]
    days_keysList = [key for key in results[colm_keysList[0]]]

    ex_post_keysList = [key for key in ex_post[colm_keysList[0]][days_keysList[0]]]
    revenues_keysList = [key for key in revenues[colm_keysList[0]][days_keysList[0]]]

    ex_post_daily = pd.DataFrame(index=days_keysList, columns=ex_post_keysList)
    revenues_daily = pd.DataFrame(index=days_keysList, columns=revenues_keysList)

    if save_results == True:
        for j in colm_keysList:
            for i in days_keysList:
                for k in ex_post_keysList:
                    ex_post_daily[k][i] = sum(ex_post[j][i][k])
                # ex_post_daily['objective'][i] = ex_post[j][i]['objective'][0]

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
