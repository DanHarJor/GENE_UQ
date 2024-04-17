#What I would like to be able to do
# import enchanted_surrogates.src as datagen

#What I have to do
import os
import sys
data_gen_path = os.path.join(os.getcwd(),'enchanted-surrogates')
print(data_gen_path)
sys.path.append(data_gen_path)
import src as datagen

if __name__ == '__main__':
    # use data gen from enchanted surrogates to make the data within the uncertainty bounds

    # train a surrogate model to perfrom well within the uncertainty bounds

    # perform UQ analysis for growthrate and sensitivity analysis.