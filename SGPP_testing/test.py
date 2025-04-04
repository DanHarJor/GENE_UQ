import pysgpp
import ast
import numpy as np

a = pysgpp.DataVector(3)
a.setAll(5)
# ar = a.array()
ar = np.array(a.array())
print(ar, ar[0])
# import sys, os
# sys.path.append('/home/djdaniel/GENE_UQ/')

# from GENE_ML.gene_ml.test_functions.max_of_many_gaussians import MaxOfManyGaussians
# import numpy as np
# import matplotlib.pyplot as plt


# num_dim = 2
# bounds = [(0,1) for i in range(num_dim)]
# num_gaussians = 2
# std=0.5
# mmg = MaxOfManyGaussians(num_dim, bounds)#num_dim, num_gaussians, bounds, mean_bounds, std_bounds, seed=10)

# mmg.specify_gaussians(means=np.array([[0.25,0.25], [0.75,0.75]]), stds = np.array([[std,std],[std,std]]))
# # mmg.plot_2d_gaussians(extra=0, grid_size=200, plot_bounds=[(0,1),(0,1)], style='contour')
# # mmg.plot_2D_of_many(which2=(0,1), style='contour')
# mmg.plot_matrix_contour()

# if num_dim == 2: 
#     fig1, ax1 = plt.subplots(1,1, figsize=(4,4))
#     fig2, ax2 = plt.subplots(1,1, figsize=(4,4))
#     mmg.plot_2d_gaussians(ax1, ax_3d=ax2)

# # fig2.tight_layout()
# fig2.savefig(fname='3d_gaussian',dpi=400)

# # import pysgpp library
# import pysgpp
# from pysgpp import BoundingBox1D, createOperationEval
# import inspect

# from scipy.stats import sobol_indices
# from scipy.stats import qmc, uniform
# from scipy.integrate import nquad

# # make uniform distributions for each input, this is used later to calculate sobel indicies
# dists = []
# for b in bounds:
#     assert b[1] > b[0]
#     dists.append(uniform(loc=b[0], scale=b[1]-b[0]))

# f = lambda x0, x1: mmg.evaluate([x0, x1])
# # f = lambda x0, x1: 16.0 * (x0-1)*x0 * (x1-1)*x1*x1

# dim = 2

# poly_basis_degree = 3
# grid = pysgpp.Grid.createPolyBoundaryGrid(dim, poly_basis_degree)

# ## Doesn't work, still samples between 0 and 1
# # for d in range(dim):
# #     bb = BoundingBox1D(*bounds[d])
# #     grid.getBoundingBox().setBoundary(d,bb)

# # grid = pysgpp.Grid.createLinearBoundaryGrid(dim)

# gridStorage = grid.getStorage()

# #create regular sparse grid, level 3
# initial_level = 2
# gridGen = grid.getGenerator()

# gridGen.regular(initial_level)

# print("number of initial grid points:    {}".format(gridStorage.getSize()))

# alpha = pysgpp.DataVector(gridStorage.getSize())
# print("length of alpha vector:           {}".format(alpha.getSize()))
# # Obtain function values and refine adaptively 5 times

# x0, x1, x0_leaf, x1_leaf = [], [], [], []

# #We don't want to run the function for every point so a wrapper function should check to see if the point has been ran and if it has return that value
# def dummy_runner(f, samples, labeled_samples=None):
#     if type(labeled_samples) != type(None):
#         # print('LABELED SAMPLES',labeled_samples.items())
#         for k, v in samples.items():
#             if k in labeled_samples.keys():
#                 samples[k] = labeled_samples[k]
#             else:
#                 samples[k] = f(*k)
#     else:
#         for k, v in samples.items():
#             samples[k] = f(*k)
#     return samples
            
# labeled_samples=None
# samples = {}
# # I want to know if there are any samples that have never been a leaf
# was_leaf = []

# samples_s = []
# is_leaf_s = []
# num_refinement_steps =20

# integral, first_order_sobel_indicies, total_order_sobel_indicies = [], [], []


# def eval(pos):
#     opEval = createOperationEval(grid)
#     pos_v = pysgpp.DataVector(len(pos))
#     for i, p in enumerate(pos):
#         pos_v[i] = float(np.round(p,3))
#     return opEval.eval(alpha, pos_v)
# def eval_many(positions):
#     # positions = np.array(positions).T
#     positions_dm = pysgpp.DataMatrix(positions)
#     opEval = pysgpp.createOperationMultipleEval(grid, positions_dm)
#     results = pysgpp.DataVector(len(positions))
#     opEval.eval(alpha, results)
#     ans = np.array([results.get(i) for i in range(len(positions))])
#     return ans
#     # print('size',results.array(results))
#     # # print('type',type(results.array()))
#     # # return results
#     # ans = []
#     # for pos in positions:
#     #     ans.append(eval(pos))
#     # out = np.array(ans)
#     # return out
# def eval_int(*pos):
#     return eval(pos)

# bounds_array = np.array(bounds).T
# space_volume = np.prod(bounds_array[1]-bounds_array[0])

# for refnum in range(num_refinement_steps):
    
#     # make samples dict
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         samples[(x0,x1)] = None
#         if i > len(was_leaf):
#             # print('APPENDING')
#             was_leaf.append(False)
#     was_leaf.append(False)
    
#     # label samples
#     samples = dummy_runner(f, samples, labeled_samples)
#     samples_s.append(samples.copy())
#     labeled_samples = samples.copy()
    
#     # set function values in alpha
#     is_leaf = []
#     positions = []
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         # print(dir(gp))
#         # break
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         positions.append([x0,x1])
#         # print('function value',f(x0,x1), type(f(x0,x1)))
#         # print('dict value',samples[(x0,x1)], type(samples[(x0,x1)]))
#         alpha[i] = samples[(x0,x1)]
#         if gp.isLeaf():
#             was_leaf[i] = True
#             is_leaf.append(True)
#         else:
#             is_leaf.append(False)     
#     is_leaf_s.append(is_leaf)
    

#     # break
#     #This next function takes the function values, old alpha and turnes them into the surpluses, new alpha
#     pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)
    
#     # get UQ quantities
#     if dim ==2:
#         grid_size = 100
#         x = np.linspace(bounds[0][0], bounds[0][1], grid_size)
#         y = np.linspace(bounds[1][0], bounds[1][1], grid_size)
#         X, Y = np.meshgrid(x, y)
#         pos = np.vstack((X.ravel(), Y.ravel())).T
#         # pos = np.array(positions)
#         Zmax = eval_many(pos)
        
#         # an integral estimate for uniform dist is the mean of the points times the volume
#         integral_estimate = np.mean(Zmax) * space_volume
#         integral.append(integral_estimate)
        
#         # Samples should come from the join probabilit distributions of the inputs
#         def approx_sobol_indices(f, samples):
#             f_values = f(samples)     
#             total_variance = np.var(f_values)
#             sobol_indices = np.zeros(dim)
#             for i in range(dim):
#                 fixed_samples = np.copy(samples.T)
#                 fixed_samples[:, i] = np.mean(samples.T[:, i])
#                 f_fixed_values = f(fixed_samples.T)
#                 sobol_indices[i] = np.var(f_fixed_values) / total_variance
#             return sobol_indices
#         first_order_sobel_indicies.append(approx_sobol_indices(eval_many, pos))        
#         # sobol = sobol_indices(func=eval_many, n=2**5, dists=dists)
#         # first_order_sobel_indicies.append(sobol.first_order)
#         # total_order_sobel_indicies.append(sobol.total_order)
#     if dim == 2 and refnum == num_refinement_steps-1 or refnum==8:
#         # print('example',eval((0.1,1.1))
#         fig, (ax,ax_3d) = plt.subplots(1,2)
#         Zmax, pos = mmg.plot_2d_gaussians(ax, ax_3d=ax_3d, grid_size=100, new_eval=eval) 
#         ax.set_aspect('equal')
#         # integrad, integrad_err = nquad(eval_int, bounds)
#         # print(integrad, integrad_err)
#         # integral.append
#     # _______________
    
#     gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, 1))
#     print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))    
#     alpha.resizeZero(gridStorage.getSize())