# class Config():
#     def __init__(self):
#         ##Parser
#         #The parsers main function is write_input_file
#         # wite_input_file takes a parameters file from base_params_path and a set of points in the form of a dict {param1:[point1,point2...], param2:[point1,point2...]...} 
#         #  It will then create a parameters file that can scan over all the points.
#         self.base_params_path = "/home/djdaniel/GENE_UQ/parameters_base_uq_highprec" 
#         self.remote_save_base_dir=f'/scratch/project_462000451/gene_out/gene_auto/'
#         self.save_dir = "/home/djdaniel/GENE_UQ/temp/"


#         ## Runner
#         #The Runner is responsible for actually running a parameters file on lumi. Its main function is code_run.
#         # code_run will take the set of points named samples and parse them into a parameters file.
#         #  It then uses ssh to run GENE with this parametres file and a passed sbatch script.
#         self.host = 'lumi1' #needs to be configured in /home/<user>/.ssh/config
#         self.sbatch_base_path = "/home/djdaniel/GENE_UQ/sbatch_base" 
#         # guess_sample_wallseconds = 200 # a guess for the number of seconds it takes to run one sample.
#         self.remote_run_dir = '/project/project_462000451/gene/'
#         self.local_run_files_dir = "/home/djdaniel/GENE_UQ/run_files"

#         #Paramiko
#         #Some SSH is done with bash commands using the .ssh/config host given above
#         #Other SSH is done with paramiko, aka what I should have used from the start.
#         #___________________
#         import paramiko
#         p_host = 'lumi.csc.fi'
#         p_user = 'danieljordan'
#         p_IdentityFile = '/home/djdaniel/.ssh/lumi-key'
#         #___________________
#         with open(p_IdentityFile, 'r') as key_file:
#             pkey = paramiko.RSAKey.from_private_key(key_file)        
#         # pkey = paramiko.RSAKey.from_private_key_file(p_IdentityFile, password=p_password)
#         self.paramiko_ssh_client = paramiko.SSHClient()
#         policy = paramiko.AutoAddPolicy()
#         self.paramiko_ssh_client.set_missing_host_key_policy(policy)
#         self.paramiko_ssh_client.connect(p_host, username=p_user, pkey=pkey)
#         self.paramiko_sftp_client = self.paramiko_ssh_client.open_sftp()

#         remote_params_dummy_file = self.paramiko_sftp_client.open('.bashrc') # any dummy file would serve this purpose
#         self.paramiko_file_type = type(remote_params_dummy_file) 
#         #___________________

# config = Config()

# if __name__ == '__main__':
#     config = Config()
#     print(config.save_dir)


    


import paramiko
from scp import SCPClient
class Config():
    def __init__(self, local=False):
        ##Parser
        #The parsers main function is write_input_file
        # wite_input_file takes a parameters file from base_params_path and a set of points in the form of a dict {param1:[point1,point2...], param2:[point1,point2...]...} 
        #  It will then create a parameters file that can scan over all the points.
        # Ensure number_of_paralell_sims = 1, as I am not sure if the geneerr.log file order will match the run numbers if the simulations are ran in paralell.
        # ensure syntax of '    var = value'. The spaces are important.  This is for parsing reasons. 

        # self.save_dir = "/home/djdaniel/DEEPlasma/temp/"
        self.base_params_path = "/home/djdaniel/GENE_UQ/parameters_base_uq_highprec" 
        self.remote_save_base_dir=f'/scratch/project_462000451/gene_out/gene_auto/'
        self.save_dir = "/home/djdaniel/GENE_UQ/temp/"




        self.local_backup_dir = '/home/djdaniel/GENE_UQ/vault/'


        ## Runner
        #The Runner is responsible for actually running a parameters file on lumi. Its main function is code_run.
        # code_run will take the set of points named samples and parse them into a parameters file.
        #  It then uses ssh to run GENE with this parametres file and a passed sbatch script.
        self.host = 'lumi' #needs to be configured in /home/<user>/.ssh/config
        self.base_sbatch_path = "/home/djdaniel/GENE_UQ/sbatch_base" 
        # single_run_timelim = 200 # a guess for the number of seconds it takes to run one sample.
        self.remote_run_dir = '/project/project_462000451/gene/'
        self.local_run_files_dir = "/home/djdaniel/GENE_UQ/run_files"
        self.local_username = 'djdaniel' # used to identify if a directory is local or on remote.

        #Paramiko
        #Some SSH is done with bash commands using the .ssh/config host given above
        #Other SSH is done with paramiko, aka what I should have used from the start.
        #___________________
        self.p_host = 'lumi.csc.fi'
        self.p_user = 'danieljordan'
        self.p_IdentityFile = '/home/djdaniel/.ssh/lumi-key'
        if not local:
            self.paramiko_setup()
    
    def paramiko_setup(self):
        with open(self.p_IdentityFile, 'r') as key_file:
            pkey = paramiko.RSAKey.from_private_key(key_file)        
        # pkey = paramiko.RSAKey.from_private_key_file(p_IdentityFile, password=p_password)
        self.paramiko_ssh_client = paramiko.SSHClient()
        policy = paramiko.AutoAddPolicy()
        self.paramiko_ssh_client.set_missing_host_key_policy(policy)
        self.paramiko_ssh_client.connect(self.p_host, username=self.p_user, pkey=pkey)
        self.paramiko_sftp_client = self.paramiko_ssh_client.open_sftp()
        self.scp = SCPClient(self.paramiko_ssh_client.get_transport())
        remote_params_dummy_file = self.paramiko_sftp_client.open('.bashrc') # any dummy file would serve this purpose
        self.paramiko_file_type = type(remote_params_dummy_file) 
        print(self.paramiko_file_type)
        #___________________
        # import os
        # import sys
        # print('Appending to path:', os.path.join(os.getcwd(),'config'))
        # sys.path.append(os.path.join(os.getcwd(),'config'))


if __name__ == '__main__':
    config = Config()
    # print(config.save_dir)


    


    