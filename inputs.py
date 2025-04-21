import numpy as np

#########################################################
# Define run setting
#########################################################

mipgap = 0.001  # Solver parameter
timelimit = 5 * 60  # Solver parameter
case = 'DK1_23'  # DK1_22, DK1_21, DK2_22, DK1_23
FCR_factor = 1  # Is the FCR reserve included? Yes or no
mFRR_factor = 0  # Is the mFRR reserve included? Yes or no

FCR_min = 2 * FCR_factor  # Minimum bid size in MW
FCR_max = 10 * FCR_factor  # Maximum bid size in MW
mFRR_up_min = 2 * mFRR_factor  # Minimum bid size in MW
mFRR_up_max = 10 * mFRR_factor  # Maximum bid size in MW
mFRR_dw_min = 2 * mFRR_factor  # Minimum bid size in MW
mFRR_dw_max = 10 * mFRR_factor # Maximum bid size in MW

Days = 13  # Number of days in analysis

save_results = True
model = ['alpha_0']  # Model versions 'alpha_1', 'alpha_0', 'alpha_r' or 'no_as' ['alpha_1', 'alpha_0', 'alpha_r', 'no_as']


#folders = ['AS_paper/Results/DK1_23_Demand_0', 'AS_paper/Results/DK1_23_Demand_500', 'AS_paper/Results/DK1_23_Demand_1000', 'AS_paper/Results/DK1_23_Demand_1500', 'AS_paper/Results/DK1_23_Demand_2500', 'AS_paper/Results/DK1_23_Demand_3000', 'AS_paper/Results/DK1_23_Demand_3500']
#H_mins = [0, 500, 1000, 1500, 2500, 3000, 3500]

folders = ['AS_paper/results_final']

#folders = ['AS_paper/Results/' + case + '/']
H_mins = [2000]
H_min = 2000
#########################################################
# Define ranges
#########################################################

dT = 1  # hourly resolution
NT = int(24 / dT)  # number of time steps
ND = 5  # number of dispensers/tube trailers
NJ = int(1)  # number of 1 hour intervals in FCR block
NI = int(NT / NJ)  # number of FCR blocks in horizon
N_down = int(2 / dT)  # number of down timesteps after ely shut of
NS = 4  # number of segments

D = range(ND)  # range over dispensers/tube trailers
T = range(NT)  # range timesteps in horizon (24 hours)
I = range(NI)  # range over FCR blocks in horizon
J = range(NJ)  # range over mFRR blocks in FCR blocks
S = range(NS)  # range over piecewise segments on ely production curve

T1 = range(1, NT)  # range over time excluding the first step

#########################################################
# Define electrolyzer parameters
#########################################################

C_E = 10  # Electrolyzer capacity in MW
E_min = 0.1  # Electrolyzer miniumum load in %

# Pieces on production curve h = A * p + B
A = np.asarray([21.15664817, 18.96085366, 16.87083602, 14.99118166])
B = np.asarray([-0.35819557, 0.19075305, 1.23576187, 2.64550265]) * C_E

P_min = np.array([0.1, 0.25001, 0.5001, 0.75001]) * C_E  # Minimum load per electrolyzer segment in MW
P_max = np.array([0.25, 0.5, 0.75, 1.]) * C_E  # Maximum load per electrolyzer segment in MW
P_sb = 0.025 * C_E  # Standby consumption electrolyzer in MW

S_max = 1000  # Maximum capacity of tube trailer in kg
C_D = 1000  # Dispenser capacity in kg

TT_bin = np.zeros((NT, ND)) + 1  # Tube trailer schedule, put to always available for simplicity

pi_H2 = 5

