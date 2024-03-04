import numpy as np
import matplotlib.pyplot as plt
import math

# UJ24-290322 at T = 3.0 J 
list_shots = [[1023 , 481 , 248,  151,  135 , 119 , 107 , 100 ,  72,   52,   46 ,  39 ,  42,   19,
   13 ,   6 ,  14 ,  11  ,  3 ,   3 ,   2 ,   3,    3],
              [728, 442, 280, 172, 109,  91 , 86 , 73,  87 , 59,  67,  55 , 56,  62,  46,  37,  30,  32,
  32,  36,  11,  14,  14,  14 , 10,   8,   7 ,  7,   6,   6,   4 ,  3,   1,   1,   1,   2,
   1 ,  1,   1],
              [457, 416, 308, 178, 144, 103,  79 , 74 , 52 , 64 , 49 , 47,  44 , 47 , 37 , 42,  42,  40,
  35 , 41,  32,  37,  44,  35 , 27,  26,  22,  20 , 13,  16,  17,   8,  11 , 12 ,  2 ,  9,
   8,   3 ,  9 ,  5,   4 ,  6 ,  4,   8 ,  3,   2,   2 ,  1,   3,   1,   0 ,  0 ,  1 ,  1,
   0 ,  0 ,  0,   1],
   [291, 331 ,277, 205 ,166, 120,  88,  77,  67,  63,  40,  50,  39,  49,  41,  39,  25,  42,
  28,  27,  24,  42,  28,  26,  25,  24,  25,  36 , 42,  29,  28,  30,  26,  20,  20,  20,
  13,  11,  12,  10 , 13,   9 ,  7 , 11,   9 ,  3,  12 ,  9 ,  5,   5,   6 ,  6,   3 ,  2,
   8,   3 ,  1 ,  8,   0 ,  0 ,  4 ,  0 ,  3 ,  1,   2 ,  0  , 1 ,  1 ,  0 ,  1 ,  0 ,  2,
   0 ,  0,   0,   0 ,  0,   0,   1],
   [168, 244 ,230 ,222, 169, 146, 101,  90,  84,  68,  48,  53,  39,  38,  33,  43,  34,  34,
  28,  28,  23,  29,  27,  28,  24,  29,  30,  21,  21,  18,  27,  12,  23 , 24,  26 , 25,
  31,  18,  22 , 21 , 29,  24,  11 , 22,  20 , 19,  13 , 10 , 11 ,  9 , 10,  11,  10,   8,
   6 ,  5  , 7 ,  6 ,  5,   6,   7 ,  8,   3,   3 ,  7 ,  6,   3 ,  5,   2 ,  2 ,  2,   1,
   3,   4,   0,   1 ,  1 ,  2,   2 ,  0 ,  1,   1,   2 ,  0,   0 ,  0 ,  1 ,  0 ,  1   ,0
,   0,   0,   3],
   [ 39,  90, 130, 151, 174, 150, 128, 104, 113,  86,  76,  62,  55  ,56 , 44,  41,  48,  34
,  38,  30,  26,  30,  19,  30,  26,  19,  22,  29,  16,  23 , 13,  24,  21,  25 , 20,  12,
  24,  20,  18,  18,  22,  19,  14,  18,  15,  17 , 17,  11,  18 , 25 , 12,  21,  22 , 17
,  22,  20,  15 , 15,  19 , 15,  13,  14,   9 , 18,  18 ,  8 , 11,   8,  11,  10,   5,   8,
   7 ,  5,   4 , 10,   5,   9,   4 ,  5 ,  4,   3 ,  0 ,  6 ,  4 ,  5 ,  9,   4,   0,   1,
   2,   1,   3 ,  8,   1,   3 ,  0,   1,   0 ,  0 ,  1 ,  1 ,  3,   1,   1 ,  1 ,  0 ,  0,
   0 ,  3 ,  0,   0 ,  1  , 1 ,  1 ,  1 ,  0,   0 ,  0 ,  1 ,  0,   1],
    [ 6 , 30 , 53,  83,  74, 114, 103, 129, 102, 101,  82 , 92,  85,  67,  60,  48,  53,  47,
  45,  45,  42,  27 , 38,  25 , 32 , 30,  27,  21 , 26,  16,  11,  17 , 23,  28 , 22,  16,
  19 , 21,  15 , 13 , 16,  17,  15 , 18 , 19 , 25 , 19,  15,  11,  15,  20,  19,  13,  21,
  18,  13,  20,  21,  13 , 14,  13,  18 , 15,  21,  11 , 20 , 21,  16,  10,  27,  16,   9,
  13,  11,  15,  13,  10,  12 , 10 , 13,   8 , 11 ,  7 ,  5 , 10,   7 ,  5,   7,   2 ,  6,
   1 ,  8,   2 ,  7 ,  6 ,  4 ,  5 , 10 ,  4  , 5 ,  5 ,  2 ,  3  , 3 ,  1  , 2 ,  3,   1,
   2,   4,   1  , 1 ,  3 ,  1  , 2 ,  0 ,  0 ,  1,   1 ,  1 ,  1 ,  0,   1 ,  0,   0,   0,
   2,   0,   0 ,  0 ,  2,   1,   0 ,  1 ,  1 ,  0 ,  1,   1],
   [ 4,  6 ,14 ,22 ,21 ,26 ,52 ,51, 53, 59, 61 ,74, 70, 79, 67, 72, 61, 70, 50, 72 ,62, 40 ,46 ,46,
 51, 43 ,43 ,23, 25, 34, 37, 37, 27, 29 ,22 ,18, 21 ,15, 22, 23 ,17, 18, 16 ,26, 15, 31, 25, 18,
 18, 13, 12, 16 ,20, 20 ,14 ,19 ,19, 13, 15 ,18 ,14 ,18, 19 ,15, 17 ,14 ,18, 16, 19, 13, 29, 18,
 18 ,16, 20 ,10 ,20, 18 , 0 ,18, 17 ,18, 12, 12, 15 ,16, 14, 13 ,12,  9 , 9,  8 ,14,  8 , 7,  9
, 11 , 6 , 9 , 7 , 8 , 1 , 9, 10 , 4 , 6  ,7 , 2 , 4  ,5 , 3  ,5  ,5  ,4  ,3  ,5  ,5  ,4  ,1 , 4
,  1 , 1 , 2 , 0  ,2 , 4  ,0 , 1 , 1 , 2 , 0 , 1 , 1 , 0 , 0 , 1 , 0 , 2 , 1,  1 , 1,  2 , 0  ,0,
  0,  0 , 1 , 0,  1 , 0 , 1 , 0 , 1 , 0 , 0 , 0,  0,  2],
  [ 2,  4 , 3 , 6 ,13 ,14, 22 ,25 ,26 ,29 ,33 ,39, 44, 45 ,50 ,61, 57 ,54, 68 ,47, 54 ,56 ,54, 46,
 57 ,50, 46, 39 ,45 , 0, 41, 40 ,46, 28 ,31 ,26, 35 ,29 ,43, 42, 19 ,30 ,23 ,24 ,22, 25 ,32, 22,
 26 ,21 ,18 ,19 ,16, 13, 13, 10, 20 ,20, 21 , 0 ,19 ,17 ,18, 19 ,12 ,23, 25 ,15,  8, 19 ,18, 27,
 18, 21, 14, 18 ,19,  9 ,13, 18 ,18, 19, 14, 20, 14, 23, 20, 16, 10 , 0 ,14, 18 ,16, 19 , 9 ,11,
 13 ,13, 12 ,11,  7 , 8 , 9 , 6 , 8 , 7, 12 , 8,  6 , 6,  1 , 8 ,10,  9,  2 , 4 , 9 , 5,  4 , 0,
  6,  4 , 8 , 6 , 4 , 0 , 2 , 6 , 3,  2 , 6  ,2 , 1 , 3,  0 , 1,  2 , 2,  1 , 0 , 2 , 2 , 0,  1,
  1 , 0,  1 , 0,  0 , 0 , 0 , 2 , 0 , 1 , 1 , 2,  2 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0,
  0,  1 , 1 , 0 , 0,  0 , 0,  0 , 0 , 0 , 1]]

# The volume sizes where we count atoms (delta k) 
sizes = [0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080, 0.10] 

# Form the lists to preserve the correlation functions and their confidence intervals
corr_funcs = []
lower_bounds_ci = []
upper_bounds_ci = []

# Calculate the correlation function and corresponding confidence interval for each size
# Correlation function g^(n) = <N*(N-1)...(N-n+1)> / <N>^n 
for shots, size in zip(list_shots, sizes):
    data = np.repeat(np.arange(len(shots)), shots)

    # Calculate the correlation function 
    # Here - the second order of the correlation function
    corr_func = np.mean((data - 1) * data) / (np.mean(data)) ** 2
    # An example for the third order (another orders are formed analogously)
    # corr_func = np.mean((data - 2) *(data - 1) * data) / (np.mean(data)) ** 3
    corr_funcs.append(corr_func)

    # Compute the bootstrap confidence interval
    N_boots = 10000  # Number of the bootstrap samples
    corr_func_boots = []
    for i in range(N_boots):
        # Choose randomly the numbers from the original sample to create the pseudo samples 
        pseudo_sample = np.random.choice(data, size=len(data), replace=True)
        # Here - the second-order correlation function
        boots_corr_func = np.mean((pseudo_sample - 1) * pseudo_sample) / (np.mean(data)) ** 2
        # An example for the third order (another orders are formed analogously)
        # boots_corr_func = np.mean((pseudo_sample - 2) * (pseudo_sample - 1) * pseudo_sample) / (np.mean(data)) ** 3
        corr_func_boots.append(boots_corr_func)
    
    # The level of confidence comprises 95%
    lower_bound_ci, upper_bound_ci = np.percentile(corr_func_boots, [2.5, 97.5])
    lower_bounds_ci.append(lower_bound_ci)
    upper_bounds_ci.append(upper_bound_ci)

# Print the correlation functions at each size with corresponding confidence intervals
#print(corr_funcs)
#print(ci_upper_bounds)
#print(ci_lower_bounds)

# Draw n-factorial line (should be changed for each new order of the correlation function)
n = 2
n_factorial = math.factorial(n)
plt.axhline(n_factorial, color = 'black', linestyle = '--', label = 'n!')

# Plot the correlation function and its confidence intervals
plt.plot(sizes, corr_funcs, marker='o', linestyle='-', label='Second-Order Correlation Function')
plt.fill_between(sizes, lower_bounds_ci, upper_bounds_ci, color='#00B2EE', alpha=0.6, label='Confidence Interval')
plt.xlabel('δk') #sizes of the volume
plt.ylabel('Second-Order Correlation Function')
plt.legend()

# Wanted to start drawing the graph at delta k = 0.02 - the lowest volume 
# So, if the smallest volume size will be changed - it should be modified as well
plt.xlim(0.020, max(sizes))  

plt.show()

