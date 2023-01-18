import os
direc = r"C:\Users\rfkov\Documents\SynapseData\LBN_Synapse_Data\awaiting_analysis"
dirs = os.listdir(direc)


dirs2 = [i.split("_")[1] for i in dirs if i not in ["mismatchLog.csv", "problem_with_h5", "set_aside_for_now"]]
dirs3 = list(set(dirs2))
len(dirs3)

errfile = r"C:\Users\rfkov\Documents\SynapseData\LBN_Synapse_Data\awaiting_analysis\problem_with_h5"

paths =  [os.path.join(direc, file) for file in os.listdir(direc)]
# filepath = r"C:\Users\rfkov\Documents\SynapseData\LBN_Synapse_Data\awaiting_analysis\LBNC_124_RI60R12-210106-094420"
import h5py
for filepath in paths:
    operantPath = r"C:\Users\rfkov\Documents\SynapseData\LBN_Synapse_Data\utility_files\OperantData.h5"
    filename = filepath.split('\\')[-1].split('-')[0]
    fileVals = filepath.split('\\')[-1].split('-')[0].split('_') #['LBN', '078', 'RI60R12S1'] extracted from file name give group names in h5 file from R with MED data
    fileKeys = ['Experiment', 'Subject', 'Paradigm']
    if fileVals[0] == 'LN' or fileVals[0] == 'LNC' or fileVals[0] == 'LBNC': #common typo in my data
        fileVals[0] = 'LBN'
    fileD = dict(zip(fileKeys, fileVals)) #{'Experiment': 'LBN', 'Subject': '078', 'Paradigm': 'RI60R12S1'}
    H5group = 'data.'+ '/'.join(fileD.values()) #'data.LBN/078/RI60R12S1'
    try:
        with h5py.File(operantPath, 'r') as f: #with is here to close everything up nicely... could just close it though
            OperantH5_dict = dict()
            for dset in f[H5group].keys():  #this will loop through all events in group, names dset     
                if any(f[H5group][dset]['value'][:]):
                    OperantH5_dict[dset] = f[H5group][dset]['value'][:] #assigns values under key to numpy array in dictionary
    except:
        os.rename(filepath, os.path.join(errfile, filename))
        

dirs2 = os.listdir(errfile)


dirs2 = [i.split("_")[1] for i in dirs]
