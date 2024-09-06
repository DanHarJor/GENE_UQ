import os
from GENE_ML.gene_ml.results.results import ResultsGroundTruthTest

from GENE_ML.gene_ml.parsers.GENEparser import GENE_scan_parser
from GENE_ML.gene_ml.dataset.ScanData import ScanData
import numpy as np
import pandas as pd
from config import config
import os

def mode_transition_test(model, name):
    nominal = [2.7954880, 5.3871083, 1.5417906]
    parameters = ['_grp_species_0-omt','_grp_species_1-omt','species-omn']
    remote_save_names = ['modeTrans-highprec-'+p for p in parameters]
    # parser
    parser = GENE_scan_parser(config.save_dir, config.base_params_path)
    GENE_groundTruth = []
    for rms in remote_save_names:
        GENE_groundTruth.append(ScanData(rms, parser=parser, host=None, remote_path=None,  test_percentage=0))

    # Putting ground truth in correct format for inference, with other nominal parameters in place -------------   
    nominal_block = np.stack([nominal for i in range(len(GENE_groundTruth[0].df))])
    nominal_df = pd.DataFrame(nominal_block)
    nominal_df.columns=parameters

    for i in range(len(parameters)):
        new_df = nominal_df.copy()
        col = GENE_groundTruth[i].df.columns.values.tolist()
        new_df[parameters[i]] = GENE_groundTruth[i].df[col[1]].tolist()
        new_df['growthrate']=GENE_groundTruth[i].df['growthrate'].tolist()
        new_df['frequency']=GENE_groundTruth[i].df['frequency'].tolist()
        new_df.insert(0, 'run_time', GENE_groundTruth[i].df['run_time'].to_numpy())
        GENE_groundTruth[i].df = new_df
        GENE_groundTruth[i].set_from_df()    
        # print(GENE_groundTruth[i].x[0:5])
        # print(parameters)
        # print('NEW DF', GENE_groundTruth[i].df.head(5))

    #--------------------------------------------------------------------------------

    from IPython.display import display
    for gt in GENE_groundTruth:
        display(gt.df.head())

    results = ResultsGroundTruthTest(name+'.resultsgt')

    if os.path.exists(results.path):
        print('\nLOADING RESULTS FROM FILE\n')
        results = results.load()
        print(f'{results.name} IS LOADED')
    else:
        results.altered_parameters_names = GENE_groundTruth[0].df.columns.values.tolist()
        for i in range(len(GENE_groundTruth)):
            print('\nCOMPUTING RESULTS\n')
            #Sampling already done, gene has been ran
            print("GT DATA NAME",GENE_groundTruth[i].name)
            if model.model_type_id == 'gpr':
                pred = model.predict(GENE_groundTruth[i].x, disclude_errors=True)
            else:
                pred = model.predict(GENE_groundTruth[i].x)
            results.growthrates.append(pred)
        results.save()

    import matplotlib.pyplot as plt
    gene_out = ['growthrate','frequency']
    width = 5
    height =5
    ncol = len(gene_out)
    nrow = len(GENE_groundTruth)
    figure, AX = plt.subplots(nrow,ncol, figsize=(ncol*width, nrow*height))

    for i in range(len(AX)):
        for j in range(len(AX[0])):
            if i == 0 and j == 0: 
                infer_label = name
                gene_label = 'GENE ground truth'
            else:
                infer_label = None
                gene_label = None

            # col = GENE_groundTruth[i].df.columns.values.tolist()
            # print(GENE_groundTruth[i].df[parameters[i]].head(2))
            x = GENE_groundTruth[i].df[parameters[i]].to_numpy(dtype=float)
            y = GENE_groundTruth[i].df[gene_out[j]].to_numpy(dtype=float)
            
            if j == 0:
                AX[i][j].plot(x, results.growthrates[i], '-g', label=infer_label)

            AX[i][j].plot(x, y, '.b', label =gene_label)
            AX[i][j].set_xlabel(parameters[i])
            AX[i][j].set_ylabel(gene_out[j])
            
    figure.tight_layout()
    figure.legend()
    plt.show(figure)