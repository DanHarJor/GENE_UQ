conda create -n uq_gene -c conda-forge -c pytorch -c anaconda optuna ipykernel scipy matplotlib emcee pymc pandas numpy scikit-learn tqdm nb_conda_kernels f90nml paramiko
conda env export --name uq_gene > uq_gene.yml
