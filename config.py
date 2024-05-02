import os
#Here is were you specify the UQ configuration

#DEEPlasma
# parameters = ['box-kymin', '_grp_species_1-omt', 'species-omn']
# bounds = [(0.05,1), (10,70), (5,60)]
# num_samples = 100
#Parser args
# base_params_path = os.path.join(os.getcwd(),'parameters_base_dp')


#UQ
parameters = ['box-kymin', '_grp_species_0-omt', '_grp_species_1-omt']
##Sampler args dependant on sampler chosen
bounds = [(0.1,100.0),(2,3.5), (4,6.75)]
# num_samples = 5
##Parser args
base_params_path = os.path.join(os.getcwd(),'parameters_base_uq')

