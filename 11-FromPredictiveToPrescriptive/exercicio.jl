
#############################################################################################################
################################### Example: Two-Stage Newsvendor Problem ###################################
#############################################################################################################

#############################################################################################################
############################################ Calling the packages ###########################################
#############################################################################################################

# Packages to the two-stage problem
using JuMP
using Gurobi
using Distributions
using Random
# Packages to the kNN
using NearestNeighbors
using Plots
using StatsPlots

#############################################################################################################
############################ Defining the coefficients and making the simulation ############################
#############################################################################################################

Random.seed!(9)
c = 10
r = 5
q = 25
u = 150

dy = 1 # locations
dx = 3 # dimension of X
# X is not iid. Instead, it follows an ARMA(2,2) model -> X = Φ1*X1 + Φ2*X2 + U + Θ1*U1 + Θ2*U2

# Matrixes with the coefficients of the ARMA(2,2) model
# Coefficients from AR part
Φ1 = [0.5 -0.9 0;
      1.1 -0.7 0;
      0    0   0.5
]

Φ2 = [ 0   -0.5 0;
      -0.5  0   0;
       0    0   0
]

# Coefficients from MA part
Θ1 = [0.4  0.8 0;
      1.1 -0.3 0;
      0    0   0
]

Θ2 = [ 0   -0.8 0;
      -1.1 -0.3 0;
       0    0   0
]

# Coefficient for the multivariated normal - the multivariate normal generates erros
ΣU = [1   0.5 0;
      0.5 1.2 0.5;
      0   0.5 0.8
]

# Coefficient for the multivariated normal
μ = [0; 0; 0]

# The last two observations (for the AR part)
X1 = [1; 1; 1]
X2 = [1; 1; 1]

# Demand D -> generated accoding to a factor model -> D = A * (X + δ/4) + (B*X) .* ϵ
A = 2.5 * [ 0.8 0.1 0.1;
            0.1 0.8 0.1;
            0.1 0.1 0.8;
            0.8 0.1 0.1;
            0.1 0.8 0.1;
            0.1 0.1 0.8;
            0.8 0.1 0.1;
            0.1 0.8 0.1;
            0.1 0.1 0.8;
            0.8 0.1 0.1;
            0.1 0.8 0.1;
            0.1 0.1 0.8
]

B = 7.5 * [  0 -1 -1;
            -1  0 -1;
            -1 -1  0;
             0 -1  1;
            -1  0  1;
            -1  1  0;
             0  1 -1;
             1  0 -1;
             1 -1  0;
             0  1  1;
             1  0  1;
             1  1  0
]

S = 1000; # Total number of scenarios