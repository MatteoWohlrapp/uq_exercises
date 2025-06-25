import chaospy as cp
import numpy as np

# TODO: DONE. Fix seed
np.random.seed(42)

if __name__ == '__main__':
	# TODO: DONE: define the two distributions
	unif_distr = cp.Uniform(-1, 1)
	norm_distr = cp.Normal(10, 1)
 
	distr = unif_distr

	# degrees of the polynomials
	N = [8,] # [2, 5, 8]  # N = [8,]

	# generate orthogonal polynomials for all N's
	for i, n in enumerate(N):
		
		# TODO: DONE: employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions
		orth_poly = cp.generate_expansion(n, distr, normed=True)

		# TODO: DONE: compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]
		for i in range(n+1):
			for j in range(n+1):
				inner_product = cp.E(orth_poly[i] * orth_poly[j], distr)
				# TODO: DONE: print result for specific n
				print(f"E[phi_{i} * phi_{j}] = {inner_product:.2f}")