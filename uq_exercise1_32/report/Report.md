# Bonus Exercise 1. Team 32. Report. Monte Carlo sampling and variance reduction techniques.

# Team Members
Diana Deldar Abdolmaleki - 03732135
Maximilian Fehrentz - <Insert Matrikel>
Matteo Wohlrapp - <Insert Matrikel>

# 1. Introduction to Monte Carlo sampling estimator

## Assignment 1.1

### Task
Assume that a vector G represents a set of exam grades for a course. First, compute the mean and the variance of G using basic Python functionality.

Then, compute the same qunatities of interest using numpy’s functions. Do you get identical resutls? If not, why?

### Our Solution
grades [1.3 1.7 1.  2.  1.3 1.7 2.  2.3 2.  1.7 1.3 1.  2.  1.7 1.7 1.3 2. ]
mean_python       1.6470588235294117, var_python       0.14889705882352938
mean_numpy_ddof_0 1.6470588235294117, var_numpy_ddof_0 0.14013840830449825
mean_numpy_ddof_1 1.6470588235294117, var_numpy_ddof_1 0.14889705882352938

We can see that the mean values are the same in all three cases - our python calculation, using the formula from the lecture, and the two cases using numpy built-in function with different parameter values for ddof (degrees of freedom) being set to 0 or 1.

We can see that the values of the computed variance are not the same in all three cases. The values are the same for our python calculation and the numpy built-in function when the ddof parameter is set to 1, because this creates the same divisor (n-ddof)=(n-1) as we have in our python calculation. In case we use ddof=0, the divisor is (n-ddof)=(n-0)=n, which leads to a different variance value.

## Assignment 1.2

### Task
Now, we can apply the same idea to estimate mean and covariance of a multivariate random variable X.

Consider a bivariate normal distribution with the given mean µ= [−0.4,1.1]**T and covariance matrix V = [[2,0.4],[0.4,1]]. First, generate N = [10,100,1000,10000] samples from the given bivariate normal distribution.

Next, use Monte Carlo sampling over the N samples to approximate the mean values and covariance matrix.

Report the mean and covariance values for an increasing number of samples.

As you know the true values of the mean and covairiance, plot the aboslute
error of your estimator. It is enough if you only report results for one mean
value and for two covariance values: one diagonal, one off-diagonal.

Additionally, plot the RMSE (eq. (1)) for the mean estimator

Can we use the same formula for the covariance estimator? Why?

### Our Solution
N = 10: mean [-0.5811197   0.78748464], covariance [[1.6687410346187148, -0.1651161666269441], [-0.1651161666269441, 0.4708837699753663]]
N = 100: mean [-0.36547733  1.06675853], covariance [[1.7595838189988084, 0.2305061347967368], [0.2305061347967368, 0.6503427537275073]]
N = 1000: mean [-0.28950557  1.10056329], covariance [[2.0178629217158717, 0.32103036138338786], [0.32103036138338786, 0.9674796948982014]]
N = 10000: mean [-0.40091235  1.11160838], covariance [[1.987918433957291, 0.36409937120276903], [0.36409937120276903, 1.0091297909341337]]

N = 10: rmse [0.40850226861288236, 0.21699856450570504]
N = 100: rmse [0.1326493052751807, 0.0806438313653008]
N = 1000: rmse [0.044920629133126264, 0.03110433562862582]
N = 10000: rmse [0.014099356134083894, 0.010045545236243444]

TODO: Insert plots (see "report/12_abs_errors.png", "report/12_rmse_errors.png")
TODO: Answer "Can we use the same formula for the covariance estimator?".

# 2. Monte Carlo integration

## Assignment 2.1

## Assignment 2.2

# 3. Improving standard Monte Carlo sampling

# 4. Monte Carlo sampling for forward propagation UQ


