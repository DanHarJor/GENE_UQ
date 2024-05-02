# GENE_UQ
Providing uncertainty quantification support for the gyrokinetic gene code. See, genecode.org

# Cloning a repo with sub modules
After the usual 
git clone

you must run 
git submodule init
git submodule update

OR use flag
git clone --recurse-submodules


# Updating Submodules From Github
Put the desired branch in the .gitmodules file or .gitconfig if only local 
git config -f .gitmodules submodule.repoName.branch branchName  
git submodule update --remote --init --recursive


# chat gbt
Updating Submodules:

    After adding a submodule with a specific branch, use git submodule update --init to initialize and fetch the submodule content.

To ensure that the submodule uses the specified remote branch, run:

git submodule update --remote

