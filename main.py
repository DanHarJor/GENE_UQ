#What I would like to be able to do
# import enchanted_surrogates.src as datagen

#What I have to do
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'enchanted-surrogates','src'))
from parsers.GENEparser import GENE_scan_parser
from samplers.static_sparse_grid import StaticSparseGrid
import paramiko
import numpy as np

from config import parameters, bounds, base_params_path#, num_samples 
from Samplers.uniform import Uniform

if __name__ == '__main__':
    # uniform_sampler = Uniform(bounds=bounds, num_samples=num_samples, parameters=parameters)
    sg_sampler = StaticSparseGrid(bounds=bounds, parameters=parameters, level=3, level_to_nodes=1)
    
    parser = GENE_scan_parser(base_params_path)
    
    
    # parser.write_input_file(uniform_sampler.samples, os.path.join(os.getcwd(),'parameter_files'), file_name='parameters_uniform_dp') 
    
    parser.write_input_file(sg_sampler.samples, os.path.join(os.getcwd(),'parameter_files'),file_name='parameters_sg_uq')



    # use data gen from enchanted surrogates to make the data within the uncertainty bounds

    # train a surrogate model to perfrom well within the uncertainty bounds

    # perform UQ analysis for growthrate and sensitivity analysis.