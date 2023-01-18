import os
import sys
import json
import time
import glob
import h5py
import warnings
from itertools import repeat
import numpy as np
import pandas as pd
from numpy import int32, uint32, uint8, uint16, float64, int64, int32, float32
import multiprocessing as mp


# functino to read tsq file
def readtsq(filepath):
    print("### Trying to read tsq file....")
    names = ('size', 'type', 'name', 'chan', 'sort_code', 'timestamp',
             'fp_loc', 'strobe', 'format', 'frequency')
    formats = (int32, int32, 'S4', uint16, uint16, float64, int64,
               float64, int32, float32)
    offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
    tsq_dtype = np.dtype({'names': names, 'formats': formats,
                          'offsets': offsets}, align=True)
    path = glob.glob(os.path.join(filepath, '*.tsq'))
    if len(path)>1:
        raise Exception('Two tsq files are present at the location.')
    elif len(path)==0:
        print("\033[1m"+"tsq file not found."+"\033[1m")
        return 0, 0
    else:
        path = path[0]
        flag = 'tsq'

    # reading tsq file
    tsq = np.fromfile(path, dtype=tsq_dtype)

    # creating dataframe of the data
    df = pd.DataFrame(tsq)

    print("Data from tsq file fetched....")
    return df, flag

# function to check if doric file exists
def check_doric(filepath):
    print("### Checking if doric file exists...")
    path = glob.glob(os.path.join(filepath, '*.csv')) + \
           glob.glob(os.path.join(filepath, '*.doric'))
    
    flag_arr = []
    for i in range(len(path)):
        ext = os.path.basename(path[i]).split('.')[-1]
        if ext=='csv':
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    df = pd.read_csv(path[i], index_col=False, dtype=float)
                except:
                    df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
                    flag = 'doric_csv'
                    flag_arr.append(flag)
        elif ext=='doric':
            flag = 'doric_doric'
            flag_arr.append(flag)
        else:
            pass

    if len(flag_arr)>1:
        raise Exception('Two doric files are present at the same location')
    if len(flag_arr)==0:
        print("\033[1m"+"Doric file not found."+"\033[1m")
        return 0
    print('Doric file found.')
    return flag_arr[0]
        

# check if a particular element is there in an array or not
def ismember(arr, element):
    res = [1 if i==element else 0 for i in arr]
    return np.asarray(res)


# function to write data to a hdf5 file
def write_hdf5(data, event, filepath, key):

    # replacing \\ or / in storenames with _ (to avoid errors while saving data)
    event = event.replace("\\","_")
    event = event.replace("/","_")

    op = os.path.join(filepath, event+'.hdf5')
    
    # if file does not exist create a new file
    if not os.path.exists(op):
        with h5py.File(op, 'w') as f:
            if type(data) is np.ndarray:
                f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
            else:
                f.create_dataset(key, data=data)

    # if file already exists, append data to it or add a new key to it
    else:
        with h5py.File(op, 'r+') as f:
            if key in list(f.keys()):
                if type(data) is np.ndarray:
                    f[key].resize(data.shape)
                    arr = f[key]
                    arr[:] = data
                else:
                    arr = f[key]
                    arr = data
            else:
                if type(data) is np.ndarray:
                    f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
                else:
                    f.create_dataset(key, data=data)


# function to read event timestamps csv file.
def import_csv(filepath, event, outputPath):
    print("\033[1m"+"Trying to read data for {} from csv file.".format(event)+"\033[0m")
    if not os.path.exists(os.path.join(filepath, event+'.csv')):
        raise Exception("\033[1m"+"No csv file found for event {}".format(event)+"\033[0m")

    df = pd.read_csv(os.path.join(filepath, event+'.csv'), index_col=False)
    data = df
    key = list(df.columns)

    if len(key)==3:
        arr1 = np.array(['timestamps', 'data', 'sampling_rate'])
        arr2 = np.char.lower(np.array(key))
        if (np.sort(arr1)==np.sort(arr2)).all()==False:
            raise Exception("\033[1m"+"Column names should be timestamps, data and sampling_rate"+"\033[0m")

    if len(key)==1:
        if key[0].lower()!='timestamps':
            raise Exception("\033[1m"+"Column name should be timestamps"+"\033[0m")

    if len(key)!=3 and len(key)!=1:
        raise Exception("\033[1m"+"Number of columns in csv file should be either three or one. Three columns if \
                                   the file is for control or signal data or one column if the file is for event TTLs."+"\033[0m")
        
    for i in range(len(key)):
        write_hdf5(data[key[i]].dropna(), event, outputPath, key[i].lower())

    print("\033[1m"+"Reading data for {} from csv file is completed.".format(event)+"\033[0m")

    return data, key

# function to save data read from tev file to hdf5 file
def save_dict_to_hdf5(S, event, outputPath):
    write_hdf5(S['storename'], event, outputPath, 'storename')
    write_hdf5(S['sampling_rate'], event, outputPath, 'sampling_rate')
    write_hdf5(S['timestamps'], event, outputPath, 'timestamps')

    write_hdf5(S['data'], event, outputPath, 'data')
    write_hdf5(S['npoints'], event, outputPath, 'npoints')
    write_hdf5(S['channels'], event, outputPath, 'channels')



# function to check event data (checking whether event timestamps belongs to same event or multiple events)
def check_data(S, filepath, event, outputPath):
    #print("Checking event storename data for creating multiple event names from single event storename...")
    new_event = event.replace("\\","")
    new_event = event.replace("/","")
    diff = np.diff(S['data'])
    arr = np.full(diff.shape[0],1)

    storesList = np.genfromtxt(os.path.join(outputPath, 'storesList.csv'), dtype='str', delimiter=',')
    
    if diff.shape[0]==0:
        return 0
    
    if S['sampling_rate']==0 and np.all(diff==diff[0])==False:
        print("\033[1m"+"Data in event {} belongs to multiple behavior".format(event)+"\033[0m")
        print("\033[1m"+"Create timestamp files for individual new event and change the stores list file."+"\033[0m")
        i_d = np.unique(S['data'])
        for i in range(i_d.shape[0]):
            new_S = dict()
            idx = np.where(S['data']==i_d[i])[0]
            new_S['timestamps'] = S['timestamps'][idx]
            new_S['storename'] = new_event+str(int(i_d[i]))
            new_S['sampling_rate'] = S['sampling_rate']
            new_S['data'] = S['data']
            new_S['npoints'] = S['npoints']
            new_S['channels'] = S['channels']
            storesList = np.concatenate((storesList, [[new_event+str(int(i_d[i]))], [new_event+'_'+str(int(i_d[i]))]]), axis=1)
            save_dict_to_hdf5(new_S, new_event+str(int(i_d[i])), outputPath)

        idx = np.where(storesList[0]==event)[0]
        storesList = np.delete(storesList,idx,axis=1)
        if not os.path.exists(os.path.join(outputPath, '.cache_storesList.csv')):
            os.rename(os.path.join(outputPath, 'storesList.csv'), os.path.join(outputPath, '.cache_storesList.csv'))
        if idx.shape[0]==0:
            pass 
        else:
            np.savetxt(os.path.join(outputPath, 'storesList.csv'), storesList, delimiter=",", fmt='%s')

            

# function to read tev file
def readtev(data, filepath, event, outputPath):

    print("Reading data for event {} ...".format(event))

    tevfilepath = glob.glob(os.path.join(filepath, '*.tev'))
    if len(tevfilepath)>1:
        raise Exception('Two tev files are present at the location.')
    else:
        tevfilepath = tevfilepath[0]


    data['name'] = np.asarray(data['name'], dtype=np.str)

    allnames = np.unique(data['name'])

    index = []
    for i in range(len(allnames)):
        length = len(np.str(allnames[i]))
        if length<4:
            index.append(i)


    allnames = np.delete(allnames, index, 0)


    eventNew = np.array(list(event))

    #print(allnames)
    #print(eventNew)
    row = ismember(data['name'], event)


    if sum(row)==0:
        print("\033[1m"+"Requested store name "+event+" not found (case-sensitive)."+"\033[0m")
        print("\033[1m"+"File contains the following TDT store names:"+"\033[0m")
        print("\033[1m"+str(allnames)+"\033[0m")
        print("\033[1m"+"TDT store name "+str(event)+" not found."+"\033[0m")
        import_csv(filepath, event, outputPath)

        return 0
        
    allIndexesWhereEventIsPresent = np.where(row==1)
    first_row = allIndexesWhereEventIsPresent[0][0]

    formatNew = data['format'][first_row]+1

    table = np.array([[0,0,0,0],
                        [0,'float',1, np.float32],
                         [0,'long', 1, np.int32],
                     [0,'short',2, np.int16], 
                     [0,'byte', 4, np.int8]])

    S = dict()

    S['storename'] = np.str(event)
    S['sampling_rate'] = data['frequency'][first_row]
    S['timestamps'] = np.asarray(data['timestamp'][allIndexesWhereEventIsPresent[0]])
    S['channels'] = np.asarray(data['chan'][allIndexesWhereEventIsPresent[0]])


    fp_loc = np.asarray(data['fp_loc'][allIndexesWhereEventIsPresent[0]])
    data_size = np.asarray(data['size'])

    if formatNew != 5:
        nsample = (data_size[first_row,]-10)*int(table[formatNew, 2])
        S['data'] = np.zeros((len(fp_loc), nsample))
        for i in range(0, len(fp_loc)):
            with open(tevfilepath, 'rb') as fp:
                fp.seek(fp_loc[i], os.SEEK_SET)
                S['data'][i,:] = np.fromfile(fp, dtype=table[formatNew, 3], count=nsample).reshape(1, nsample, order='F')
                #S['data'] = S['data'].swapaxes()
        S['npoints'] = nsample
    else:
        S['data'] = np.asarray(data['strobe'][allIndexesWhereEventIsPresent[0]])
        S['npoints'] = 1
        S['channels'] = np.tile(1, (S['data'].shape[0],))


    S['data'] = (S['data'].T).reshape(-1, order='F')
    
    save_dict_to_hdf5(S, event, outputPath)
    
    check_data(S, filepath, event, outputPath)

    print("Data for event {} fetched and stored.".format(event))


# function to execute readtev function using multiprocessing to make it faster
def execute_readtev(data, filepath, event, outputPath, numProcesses=mp.cpu_count()):

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
    #p = mp.Pool(mp.cpu_count())
    #p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
    #p.close()
    #p.join()
    print("Time taken = {0:.5f}".format(time.time() - start))


def execute_import_csv(filepath, event, outputPath, numProcesses=mp.cpu_count()):
    #print("Reading data for event {} ...".format(event))

    start = time.time()
    with mp.Pool(numProcesses) as p:
        p.starmap(import_csv, zip(repeat(filepath), event, repeat(outputPath)))
    print("Time taken = {0:.5f}".format(time.time() - start))

def execute_import_doric(filepath, storesList, flag, outputPath):
    
    if flag=='doric_csv':
        path = glob.glob(os.path.join(filepath, '*.csv'))
        if len(path)>1:
            raise Exception('More than one Doric csv file present at the location')
        else:
            df = pd.read_csv(path[0], header=1, index_col=False)
            df = df.dropna(axis=1, how='all')
            df = df.dropna(axis=0, how='any')
            df['Time(s)'] = df['Time(s)'] - df['Time(s)'].to_numpy()[0]    
            for i in range(storesList.shape[1]):
                if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
                    timestamps = np.array(df['Time(s)'])
                    sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
                    write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
                    write_hdf5(df['Time(s)'].to_numpy(), storesList[0,i], outputPath, 'timestamps')
                    write_hdf5(df[storesList[0,i]].to_numpy(), storesList[0,i], outputPath, 'data')
                else:
                    ttl = df[storesList[0,i]]
                    indices = np.where(ttl<=0)[0]
                    diff_indices = np.where(np.diff(indices)>1)[0]
                    write_hdf5(df['Time(s)'][indices[diff_indices]+1].to_numpy(), storesList[0,i], outputPath, 'timestamps')
    else:
        path = glob.glob(os.path.join(filepath, '*.doric'))
        if len(path)>1:
            raise Exception('More than one Doric file present at the location')
        else:
            with h5py.File(path[0], 'r') as f:
                keys = list(f['Traces']['Console'].keys())
                for i in range(storesList.shape[1]):
                    if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
                        timestamps = np.array(f['Traces']['Console']['Time(s)']['Console_time(s)'])
                        sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
                        data = np.array(f['Traces']['Console'][storesList[0,i]][storesList[0,i]])
                        write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
                        write_hdf5(timestamps, storesList[0,i], outputPath, 'timestamps')
                        write_hdf5(data, storesList[0,i], outputPath, 'data')
                    else:
                        timestamps = np.array(f['Traces']['Console']['Time(s)']['Console_time(s)'])
                        ttl = np.array(f['Traces']['Console'][storesList[0,i]][storesList[0,i]])
                        indices = np.where(ttl<=0)[0]
                        diff_indices = np.where(np.diff(indices)>1)[0]
                        write_hdf5(timestamps[indices[diff_indices]+1], storesList[0,i], outputPath, 'timestamps')

#RK script added
def CheckMED(inputParameters, op, filepath, data, storesList):
    #RK testing - run the above to def everything
    import re #will need later to exclude signals from events
    inputDir = os.path.abspath(os.path.join(filepath, os.pardir))

    if os.path.exists(os.path.join(op, '.cache_storesList.csv')):
         storesList_all = np.genfromtxt(os.path.join(op, '.cache_storesList_All.csv'), dtype='str', delimiter=',')
    else:
         storesList_all = np.genfromtxt(os.path.join(op, 'storesList_All.csv'), dtype='str', delimiter=',')
    #event = np.unique(storesList[0,:])[6]


    #pulling in operant data from R repository for a given animal
    operantPath = r"C:\Users\rfkov\Documents\SynapseData\LBN_Synapse_Data\utility_files\OperantData.h5"
    fileVals = filepath.split('\\')[-1].split('-')[0].split('_') #['LBN', '078', 'RI60R12S1'] extracted from file name give group names in h5 file from R with MED data
    fileKeys = ['Experiment', 'Subject', 'Paradigm']
    if fileVals[0] == 'LN' or fileVals[0] == 'LNC' or fileVals[0] == 'LBNC' or fileVals[0] == 'LNBC': #common typo in my data
        fileVals[0] = 'LBN'
    fileD = dict(zip(fileKeys, fileVals)) #{'Experiment': 'LBN', 'Subject': '078', 'Paradigm': 'RI60R12S1'}
    H5group = 'data.'+ '/'.join(fileD.values()) #'data.LBN/078/RI60R12S1'
    with h5py.File(operantPath, 'r') as f: #with is here to close everything up nicely... could just close it though
        OperantH5_dict = dict()
        for dset in f[H5group].keys():  #this will loop through all events in group, names dset     
            if any(f[H5group][dset]['value'][:]):
                OperantH5_dict[dset] = f[H5group][dset]['value'][:] #assigns values under key to numpy array in dictionary




    #finds events from StoreList and create _trim version of operant dictionary (form R) for comparison to Syanpse
    #contains only events from original synapse file. storesList was generated from the csv so should summon accurate pairs from python and R
    pattern = 'control.*|signal.*' #for weeding out control/signal files
    #first set ind_events and eventsList used for making pythonH5_dict based on storeList
    ind_events = [i for i, el in enumerate(storesList[1]) if not re.match(pattern, el)] #index of events (i.e. not control|signal)
    eventsList = storesList[:,ind_events] 
    #second _all versions used to call in MED data which might have more events than Synapse if there was a malfunction or human error in TTL setup
    ind_events_all = [i for i, el in enumerate(storesList_all[1]) if not re.match(pattern, el)] #index of events (i.e. not control|signal)
    eventsList_all = storesList_all[:,ind_events_all] #used below making pythonH5_dict
    OperantH5_dict_trim = {k: v for k, v in OperantH5_dict.items() if k in eventsList_all} #this is dictionary comprehension, .items returns key and value, opens with k: v which is how dictionaries are written

    #made a function for readability... probably a better way to do this...
    def switch_eventNames(eventsList_all, inputString):
        event_ind= eventsList_all[1][:].tolist().index(inputString)
        return eventsList_all[0][:].tolist()[event_ind]
    #this is making a new dictionary from MED data with old names swapped in to make comparison easier later... 
    #we need the old names for things to get assigned because preprocess.py will do the conversion for us later and we don't want to confuse it
    OperantH5_dict_trim_oldname = {switch_eventNames(eventsList_all, k): v for k, v in OperantH5_dict_trim.items()}

    # function to read hdf5 file from preprocess.py
    def read_hdf5(event, filepath, key):
         if event:
             event = event.replace("\\","_")
             event = event.replace("/","_")
             op = os.path.join(filepath, event+'.hdf5')
         else:
             op = filepath

         if os.path.exists(op):
             with h5py.File(op, 'r') as f:
                 arr = np.asarray(f[key])
         else:
             raise Exception('{}.hdf5 file does not exist'.format(event))

         return arr

    def findSimilar(a, b, tol = 0.1, AnotB = False):
        similar = list()
        if AnotB == False: #just do normal matching, can match multiple times (think matching diffs)
            for i1, va in enumerate(a): #i is index, v is value
                for i2, vb in enumerate(b): 
                    if vb-tol < va <= vb + tol:
                        similar.append(i1)
            return np.asarray(similar, dtype = int64)
        else: 
            for i1, va in enumerate(a): #no longer matching multiple times (think timestamp check)
                check = 0
                for i2, vb in enumerate(b): 
                    if vb-tol < va <= vb + tol:
                        #similar.append(i1)
                        check = 1
                        break
                if check == 0:
                    similar.append(i1)
            return np.asarray(similar, dtype = int64)

    def findSimilar2(a, b, tol = 0.1):
        similar = list()
        similarB = list()
        for i1, va in enumerate(a): #i is index, v is value
            for i2, vb in enumerate(b): 
                if vb-tol < va <= vb + tol:
                    similar.append(i1)
                    similarB.append(i2)
        return np.asarray(similar, dtype = int64), np.asarray(similarB, dtype = int64)

    def longest_conseq_matching(a, b, tol = 0.1):  #longest consecutive sequence - LCM. Returns values, indices
        #importantly, this function takes timestamps as inputs and returns the values and indices of matching values with indices increasing by 1
        #you can omit one or the other return values using longest_conseq_matching(a,b)[0|1]
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)] #make table rows of a, cols of b
        l = 0
        li = 0
        for i, ca in enumerate(a, 1): #i is index, ca are valueus, index starts at 1 rather than 0... for every row
            for j, cb in enumerate(b, 1):#same, but for every col in a row
                if cb - tol < ca <= cb + tol: #are they equal at a particular index within 0.1 tolerance?
                    table[i][j] = table[i - 1][j - 1] + 1 #uses indices from enumerate and takes l from prior element of sequence and adds 1
                    if table[i][j] > l:
                        l = table[i][j]
                        li = i
        
        table_seq = a[li-(l):li]
        table_seq_ind = list(range(li-(l),li))
        return np.asarray(table_seq), np.asarray(table_seq_ind,dtype = int64)

    #error log in csv format
    import csv
    from datetime import datetime
    def write_csv_mismatch(mismatch, H5group, inputDir, h5file = 'mismatchLog', benign = True, mismatchEventMED = "", mismatchEventSyn = "", ReplacedTotal = np.nan, KeptTotal = np.nan, Total = np.nan, propReplaced = np.nan):
        #mismatch = 'an_explicit_issue', H5group ='data.LBN/078_RI60R12S1'; h5file = "mismatchLog",
        #filepath - inputDir
         # replacing \\ or / in storenames with _ (to avoid errors while saving data)
        H5group_split = H5group.split("/")
        mismatch_dict = {"Issue": mismatch,
                         "DataSet": H5group_split[0],
                         "EventMED": mismatchEventMED,
                         "EventSyn": mismatchEventSyn,
                          "Subject": H5group_split[1],
                          "Paradigm": H5group_split[2],
                          "Benign": benign,
                          "Date": datetime.now().strftime("%m/%d/%Y"),
                          "Time": datetime.now().strftime("%H:%M:%S"),
                          "ReplacedTotalTs": ReplacedTotal,
                          "KeptTotalTs": KeptTotal,
                          "TotalTs": Total, 
                          "propReplacedTs": propReplaced}   
        header = list(mismatch_dict.keys())    
        file_exists = os.path.exists(os.path.join(inputDir, h5file+'.csv'))
        with open (os.path.join(inputDir, h5file+'.csv') ,'a') as filedata: 
            # header = ["Issue", "DataSet", "Subject", "Paradigm"]                           
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=header, lineterminator='\n')
            if not file_exists:
                writer.writeheader()
            writer.writerow(mismatch_dict)
             



    #read python-generated h5 files into similar dictionary - if statement always prefers original if matching already occurred
    #events are saved at line 247 in this script
    #for each event with original naming (NdNP...)
    pythonH5_dict = dict()
    for i in eventsList[0]:
        #could add line here to loop through output files, but for now I'm counting on only having 1
        storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
        filepathO = storesListPath[0]
        if  os.path.exists(os.path.join(op, i + '_preMEDmatch' +'.hdf5')):
            arr = read_hdf5(i + '_preMEDmatch', filepathO, 'timestamps')
        else:
            arr = read_hdf5(i, filepathO, 'timestamps')
        pythonH5_dict[i] = arr

    #might end up converting dictionaries to lists... but let's see, try importing matlab logic flow
    #OperantH5_dict_trim and pythonH5_dict for comparison
    #goal is to iterate through and find matches... irrespective of names, then assign original names... so let's change
    #Operanth5_dict_trim to original names under OperantH5_dict_trim_oldname

    #this is similar to matlab rounding, but does not correct for floating point errors such that 2.05 still rounds to 2.0 because it is really 2.049999999999... in the background
    dig = 2
    myround = np.vectorize(lambda x: round(x, dig))
    out = np.zeros((len(pythonH5_dict),len(OperantH5_dict_trim_oldname)), order='F')
    #first criteria is timestamps has to be greater than 1 to perform a match, otherwise difference between timestamps cannot be assessed
    pythonH5_dict_rename = dict()
    pythonH5_dict_timestampsConsec = dict()
    FirstMatchTs = np.full((7, len(pythonH5_dict)), np.nan, order = 'F')
    for i, k_p in enumerate(pythonH5_dict.keys()): #iterating through Synapse record leaves 0s if missing
        #out2 = zeros(4,6); will contain beginning and end of longest matching set of monotonically increasing by one index (consecutive) matches 
        #(rows1:2 is M, 3:4 is R, cols are possible events taken from R)
        #score system: 1.1 is perfect, 1 - perfect with tolerance, 2 - % M in R, 3 - None match (score 0), 4 - can't be matched size <= 1 (-1), 5 - can't be matched no field (NaN)
        out2 = np.zeros((4,len(OperantH5_dict_trim_oldname)), order='F') #does out2 contain scores, , dtype='int64'
        list_k_r = list() #this will gather keys in order that they are called to match to arrays/lists of scores/matches
        for i1, k_r in enumerate(OperantH5_dict_trim_oldname.keys()):
            list_k_r.append(k_r)
            #prior if statement asked - are events represented here... but this time I'm iterating based on what is there 
            #still need if statement to check if the length is greater than 1
            if len(pythonH5_dict[k_p]) > 1 and len(OperantH5_dict_trim_oldname[k_r]) > 1: #if there are multiple timestamps to compare
                #get timestamps for comparison
                
                M = pythonH5_dict[k_p]
                R = OperantH5_dict_trim_oldname[k_r]
                tempDiffM = myround(np.diff(M))
                tempDiffR = myround(np.diff(R))
                similarMR = findSimilar(tempDiffM, tempDiffR) #provides indexs of match within 0.1 tolerance (default, tol = 0.1).
                # similarMR, similarMR2 = findSimilar2(tempDiffM, tempDiffR) #these can be used to find indices of matches should that be needed
                # similarRM, similarRM2 = findSimilar2(tempDiffR, tempDiffM)
                if np.array_equal(tempDiffR, tempDiffM): #if the match is perfect
                    out[i, i1] = 1.1
                    # matchesM, match_indM = longest_conseq_matching(tempDiffM, tempDiffR) #examples to show howo to get values out... I'll be using indices mostly so isolating those
                    # matchesR, match_indR = longest_conseq_matching(tempDiffR, tempDiffM)
                    match_indM = longest_conseq_matching(tempDiffM, tempDiffR)[1] #see function for differences from matlab 
                    match_indR = longest_conseq_matching(tempDiffR, tempDiffM)[1] #also note for comparison, these are SeqMatches from matlab code
                    if match_indM.size == 0 or match_indR.size == 0:
                        out[i, i1] = -1;
                        out2[0:5, i1:i1+1] = np.full((4, 1), np.nan, order = 'F')
                    else:
                        out2[0:2, i1:i1+1] = np.array([match_indM[0], match_indM[-1]]).reshape(2,1, order = 'F')
                        out2[2:4, i1:i1+1] = np.array([match_indR[0], match_indR[-1]]).reshape(2,1, order = 'F')
                elif any(similarMR): #if there are matches, but not all are matches
                    match_indM = longest_conseq_matching(tempDiffM, tempDiffR)[1] #see function for differences from matlab 
                    match_indR = longest_conseq_matching(tempDiffR, tempDiffM)[1] #also note for comparison, these are SeqMatches from matlab code
                    #if either ends up empty due to rounding error - default to no matches
                    if match_indM.size == 0 or match_indR.size == 0:
                        out[i, i1] = -1;
                        out2[0:5, i1:i1+1] = np.full((4, 1), np.nan, order = 'F')
                    else:
                        out2[0:2, i1:i1+1] = np.array([match_indM[0], match_indM[-1]]).reshape(2,1, order = 'F')
                        out2[2:4, i1:i1+1] = np.array([match_indR[0], match_indR[-1]]).reshape(2,1, order = 'F')
                        out[i, i1] = len(match_indM)/len(tempDiffM)
                else: #if there are no matches
                    out[i, i1] = 0;
                    out2[0:5, i1:i1+1] = np.full((4, 1), np.nan, order = 'F')
            elif len(pythonH5_dict[k_p]) == 1 or len(OperantH5_dict_trim_oldname[k_r]) == 1: #if there is only one timestamp in one of em, can't compare diffs
                out[i, i1] = -1;
                out2[0:5, i1:i1+1] = np.full((4, 1), np.nan, order = 'F')
            else:
                out[i, i1] = float("nan");
                out2[0:5, i1:i1+1] = np.full((4, 1), np.nan, order = 'F')
                
        #here we have gone through all R for a given M, find best match based on longest string of consecutive matches, then tiebreak based on score
        #score is nice because it shows a proportion, but from manually checking, longest streak is more persuasive and reliable given that some proportions represent low n
        #out is score, out2 (row2 - row1)+1 is length of streak
        outlist = list(out[i, :])
        score = max(outlist) #like matlab max will take first entry only
        #score_index = outlist.index(score) #not used
        score_indices = np.where(outlist == score)[0]
        match_count = np.array(out2[1,:]-out2[0,:]+1, dtype=int64) #when counting matches, adding 1 to get true count rather than index subtraction: [0:36] has 37 elements, [0:0] has 1
        match_count_indices = np.where(match_count == max(match_count))[0]
        #default to use longest matching count, if multiple of those, use score to tiebreak
        if len(match_count_indices)>1 and score > 0: #if there are more than one with best num consecutive matching
            sub_outlist = list([outlist[x] for x in list(match_count_indices)]) #find any best scores that overlap with best num consecutive matching
            sub_outlist_ind = list([ix for ix, x in enumerate(list(match_count_indices))]) 
            match_count_indices[sub_outlist_ind] #this just gets the indices of the ties that correspond to events
            #again, these are the best R matches for a given M... essentially, which R record to use?
            #about to ask... are all the best matches equal? If so, just pick one, it doesn't matter
            are_not_equal = True
            for ix in match_count_indices:
                for ix2 in match_count_indices: 
                    if np.array_equal(OperantH5_dict_trim_oldname[list_k_r[ix]], OperantH5_dict_trim_oldname[list_k_r[ix2]]):
                        are_not_equal = False
            if any(sub_outlist): #if there are any scores present
                sub_outlist_max= max(sub_outlist) #select best matching 
                sub_outlist_max_all = [x for x in sub_outlist if x == sub_outlist_max]
                
                if len(sub_outlist_max_all) > 1 and are_not_equal: #see error
                    mismatch = 'Multiple Best matches... cannot decide'
                    write_csv_mismatch(mismatch, H5group, inputDir, benign = False, mismatchEventSyn=k_p)
                    print(k_p + ' in Synapse...')
                    print('MED names corresponding to Matches/Scores:')
                    print(', '.join(list_k_r))
                    print('Count of most consecutive matchinig timestamp diffs:')
                    print(match_count)
                    print('Scores (proportion matching):')
                    print(out[i, :])
                    print('Remove h5 event file if matching is really bad')
                    raise ValueError(mismatch)
                else:
                    tiebreak = sub_outlist.index(sub_outlist_max) #find index of best score among best num consecutive matching
                    best_match = match_count_indices[tiebreak] #select from best num consecutive matching using that index
            else: #something is wrong, no scores?
                mismatch = 'Multiple Best matches... none that match scores?'
                write_csv_mismatch(mismatch, H5group, inputDir, benign = False, mismatchEventSyn=k_p)
                print(k_p + ' in Synapse...')
                print('MED names corresponding to Matches/Scores:')
                print(', '.join(list_k_r))
                print('Count of most consecutive matchinig timestamp diffs:')
                print(match_count)
                print('Scores (proportion matching):')
                print(out[i, :])
                print('Remove h5 event file if matching is really bad')
                raise ValueError(mismatch)
        elif len(match_count_indices)==1 and not np.isnan(match_count[match_count_indices]) and score > 0: #if there is only one best by consec match and it's real
            best_match = match_count_indices
        else: #could be no matches, could be -1
            best_match = [np.nan]        # mismatch = 'Multiple Best matches... no scores for tie-break 2'
            # write_csv_mismatch(mismatch, H5group, inputDir, benign = False)
            # raise ValueError(mismatch)
        if hasattr(best_match, '__len__'): #accounts for but above sometimes spitting scalar
            best_match = best_match[0]
        #previously also checked to make sure best match also had best score... could be a nice sanity check?
        if not best_match in score_indices and not np.isnan(best_match):
            mismatch = 'Best match (most consecutive matches) does not have best score... investigate'
            write_csv_mismatch(mismatch, H5group, inputDir, benign = False, mismatchEventSyn=k_p)
            print(k_p + ' in Synapse...')
            print('MED names corresponding to Matches/Scores:')
            print(', '.join(list_k_r))
            print('Count of most consecutive matchinig timestamp diffs:')
            print(match_count)
            print('Scores (proportion matching):')
            print(out[i, :])
            print('Remove h5 event file if matching is really bad')
            raise ValueError(mismatch)
 
        if score > 0: #could also use best match, but we confirm these are the same immediately above
            #now we have the best match for M, we will take data for M and name it based on best match in R
            matched_event_name = list_k_r[best_match] #to R
            pythonH5_dict_rename[matched_event_name] = M
            #next 4 lines resummon prior calculations needed
            R = OperantH5_dict_trim_oldname[matched_event_name] #resummon matching R data
            tempDiffR = myround(np.diff(R)) #tempDiffM is still valid
            similarMR = findSimilar(tempDiffM, tempDiffR) #will I use this?
            match_indM = longest_conseq_matching(tempDiffM, tempDiffR)[1] #see function for differences from matlab 
            Mrange = range(out2[0, best_match].astype(int), out2[1, best_match].astype(int)+2) #+1 because we are moving from diffs to originals, then +1 again because the way python indexes ranges is the way it is
            M2 = M[Mrange] #
            pythonH5_dict_timestampsConsec[matched_event_name] = M2 #for comparison this is EventRep variable from matlab
            FirstMatchTs[0:8, i] = [R[out2[2, best_match].astype(int)],
                                                out2[2, best_match].astype(int),
                                                out2[0, best_match].astype(int),
                                                M[out2[0, best_match].astype(int)],
                                                len(match_indM)+1-len(M2),
                                                score,
                                                len(M2)]
            # FirstMatchTs(1:7, i) = [R(out2(3,ind)) out2(3,ind) out2(1, ind) M(out2(1, ind)) (length(matchIndsM)+1-length(M2)) score length(M2)]; 
            # %using the index from Diff will get you the first of the pair in original timestamps vector
            # FirstMatchTs 
            # row 1 is the first matching ts value from R, ...
            # row 2 is index within that R event vector of first match,...  
            # row 3 is same for Matlab/Synaps, ...
            # row 4 is ts values from M,...
            # row 5 is number of discarded matches
            # row 6 is score
            # row 7 is length of best match
        else:
            FirstMatchTs[5, i] = score #log score but leave rest NaN... failing to log timestampsConsec entry excludes original timestamps from matching below


    #thus ends the first major loop to collect and compare differences among timestamps
    #the ouput is FirstMatchTs which is an array where each column correponds to various important values for 
    #score comparison and data alignment (if using first verifiable matching Ts)

    if np.amax(out) < 1:
        print(" ".join(fileVals) +  "- No perfect MED to Syn match available, see mismatchLog log:")
        print(os.path.join(inputDir, "mismatchLog.hdf5"))
        print('MED names corresponding to Matches/Scores:')
        print(', '.join(list_k_r))
        print('Count of most consecutive matchinig timestamp diffs:')
        print(np.ndarray.tolist(FirstMatchTs[5,:]))
        print('Scores (proportion matching):')
        print(np.ndarray.tolist(FirstMatchTs[6,:]))
        mismatch = 'No perfect MED to Syn match available'
        write_csv_mismatch(mismatch, H5group, inputDir)

    #now find earlies timestamp to adjust things to (in Syn record because GUPPY will adjust later)
    if any(FirstMatchTs[6,:] > 2): #if there's more than 1 consecutive match
        concInd = np.where(FirstMatchTs[6,:] > 1)[0] #find where there are multiple matches
        concMaxScr = np.where(FirstMatchTs[5] == max(FirstMatchTs[5, concInd]))[0] #find scores where there are matches (scores: proportion of matching diffs in best streak to all diffs)
        BestMatchesInd = np.where(FirstMatchTs[6, concMaxScr] == max(FirstMatchTs[6, concMaxScr])) #get index of longest streak from within concMaxScr
        BestMatches = concMaxScr[BestMatchesInd] #withdraw BestMatches from concMaxScr...
        valInd = min(np.where(FirstMatchTs[0,:] == min(FirstMatchTs[0, BestMatches]))) #in case of ties, finds earliest timestamp among best scoring witih longest matching streak
        val = min(FirstMatchTs[0, valInd]) #in case of ties... could have just used first element too [0]
        if valInd.size > 1: #if more than one, that is tiebroken by val time...
            x = list(FirstMatchTs[0, valInd])
            x.index(val)
            ind_c = x.index(val)
            valInd = valInd[ind_c]
    elif any(FirstMatchTs[5,:] >= 1): #if there aren't many matches, we have low confidence so we ask if the only possible match is actually 100%
        concMaxScr = np.where(FirstMatchTs[5] == max(FirstMatchTs[5,:]))[0] #want index so can't just use max
        BestMatchesInd = np.where(FirstMatchTs[6, concMaxScr] == max(FirstMatchTs[6, concMaxScr])) #get index of longest streak from within concMaxScr
        BestMatches = concMaxScr[BestMatchesInd] #withdraw BestMatches from concMaxScr...
        valInd = min(np.where(FirstMatchTs[0,:] == min(FirstMatchTs[0, BestMatches]))) #in case of ties, finds earliest timestamp among best scoring witih longest matching streak
        val = min(FirstMatchTs[0, valInd]) #in case of ties... could have just used first element too [0]
        if valInd.size > 1: #if more than one, that is tiebroken by val time...
            x = list(FirstMatchTs[0, valInd])
            x.index(val)
            ind_c = x.index(val)
            valInd = valInd[ind_c]
    else:
        mismatch = 'No good matches to time-sync MED record'
        write_csv_mismatch(mismatch, H5group, inputDir, benign = False)
        
        raise ValueError(mismatch)

    adjMby = FirstMatchTs[3,valInd] - val #subtract this value from M to get R, or add to R to get M

    OperantH5_dict_TimeMatch = dict()
    pythonH5_dict_all = dict()
    for i1, k_r in enumerate(OperantH5_dict_trim_oldname.keys()):
        #there's a check here to see if an element of pythonH5_dict was there to begin with... skipping but keep in mind
        #the key phrase is "so it was there when we started"
        #in this case, we are not adjusting Syn to R, but the reverse so I will first step through the R script to make a new version that is shifted
        OperantH5_dict_TimeMatch[k_r] = OperantH5_dict_trim_oldname[k_r] + adjMby
        #there is some redundancy here but it's a little tricky to untangle so leaving for now (see MatP->pythonH5_dict_all)
        #essentially was this R data in Synapse at all to begin with? Could it have generated a match based on length? Could it have matched with something in R based on score?
        # took this out, but it should be redundant with rename... things only get renamed if score > 0... i1 tracks k_r and sometimes there's more operant data than synapse... #and FirstMatchTs[5, i1] > 0:
        
        if (k_r in pythonH5_dict_rename.keys()) and len(pythonH5_dict_rename[k_r]) > 1: 
            #check that consecutively accurate differences between adjacent ts correspond to actual values after alignment
                #this CAN lead to replacement of original Ts's from verified matches so I will expand the error rate in findSimilar() since diffs match closely
                #basically this is designed to catch egregious mismatches and I want to make it harder to make replacements of data from verified matching diffs
            MatP = pythonH5_dict_timestampsConsec[k_r] #pull matched timestamps
            R = OperantH5_dict_TimeMatch[k_r] #grab time matched/shifted R data
            MatP = MatP[findSimilar(MatP, R, 0.25)] #this is data you can be sure of... it is composed of matching consecutive timestamps (verified by diffs)
            Rrep = R[findSimilar(R, MatP, 0.25, AnotB = True)] #AnotB switches to find NOT similar: 
                #If you wanted to switch strategies of aligning data to nearest verified timestamp,
                #you would use all timestamps in python_dict_timestampsConsec across all events and find the nearest possible match to diff from
                #rather than use OperantH5_dict_TimeMatch which uses the earliest verified timepoint from python_dict_timestampsConsec to align records
                #this would be more a problem for very long recordings with more aggregious errors in alignment
            ReplacedTotal = len(Rrep)
            KeptTotal = len(MatP)
            Total = ReplacedTotal + KeptTotal
            propReplaced = ReplacedTotal/(Total)
            if not any(MatP):
                print(k_r + ' - full MED Event Replacement')
                mismatch = 'Full MED Event Replacement'
                write_csv_mismatch(mismatch, H5group, inputDir, mismatchEventMED=k_r, ReplacedTotal = ReplacedTotal, KeptTotal = KeptTotal, Total = Total, propReplaced = propReplaced)
            elif any(Rrep):
                print(k_r + ' - '  '  Replacement')
                mismatch = 'Some MED Event Replacement'
                write_csv_mismatch(mismatch, H5group, inputDir, mismatchEventMED=k_r, ReplacedTotal = ReplacedTotal, KeptTotal = KeptTotal, Total = Total, propReplaced = propReplaced)
            pythonH5_dict_all[k_r] =  np.sort(np.append(MatP, Rrep))
            
        elif len(OperantH5_dict_TimeMatch[k_r]) == 1: #if only 1 timestamp... only difference is where data is pulled from (not verified by diffs)
            #modified to screen data for likely single match and take it if within tolerance
            R = OperantH5_dict_TimeMatch[k_r]
            lengths = [len(v) for v in pythonH5_dict.values()] #single digits can't be confirmed so let's check if there is a Synapse entry that only has one number and test that one
            keys = list(pythonH5_dict.keys())
            if 1 in lengths:
                indices_1 = [i2 for i2, x in enumerate(lengths) if x == 1]
                choices = list()
                for i2 in indices_1:
                    k_p_s = keys[i2]
                    MatP = pythonH5_dict[k_p_s] #pull timestamps, should only be one in this case
                    choices.append(MatP[0])
                diffs = choices-OperantH5_dict_TimeMatch[k_r][0]
                MatP = np.array(choices[np.argmin(diffs)])
                Rrep = np.empty((0,0)) 
                if not R[0]-.15 <= MatP < R[0]+.15:
                    MatP = np.empty((0,0))   
                    Rrep = R
            else: 
                MatP = np.empty((0,0))
                Rrep = R #AnotB switches to find NOT similar
            #note, findSimilar error is lower than above because these are based solely on timestamp similarity, not diffs
             #R = OperantH5_dict_TimeMatch[k_r] #grab time matched/shifted R data
             # choices.append(MatP[findSimilar(MatP, R, 0.1)])
            ReplacedTotal = Rrep.size
            KeptTotal = MatP.size
            Total = ReplacedTotal + KeptTotal
            propReplaced = ReplacedTotal/(Total)
            if MatP.size == 0:
                print(k_r + ' - full MED Event Replacement')
                mismatch = 'Full MED Event Replacement'
                write_csv_mismatch(mismatch, H5group, inputDir, mismatchEventMED=k_r, ReplacedTotal = ReplacedTotal, KeptTotal = KeptTotal, Total = Total, propReplaced = propReplaced)
            elif Rrep.size == 0:
                print(k_r + ' - '  '  Replacement')
                mismatch = 'Some MED Event Replacement'
                write_csv_mismatch(mismatch, H5group, inputDir, mismatchEventMED=k_r, ReplacedTotal = ReplacedTotal, KeptTotal = KeptTotal, Total = Total, propReplaced = propReplaced)
            pythonH5_dict_all[k_r] =  np.sort(np.append(MatP, Rrep))
            
        elif not (k_r in pythonH5_dict.keys()): #last ditch effort, if Synapse record is completely missing
            #for some reason matlab still tested for possible Synapse data in matlab here... but should be ruled out by above...
            R = OperantH5_dict_TimeMatch[k_r] #grab time matched/shifted R data
            ReplacedTotal = len(R)
            KeptTotal = 0
            Total = ReplacedTotal + KeptTotal
            propReplaced = ReplacedTotal/(Total)
            pythonH5_dict_all[k_r] = R
            print(k_r + ' - full MED Event Replacement')
            mismatch = 'Full MED Event Replacement of ' + k_r
            write_csv_mismatch(mismatch, H5group, inputDir, mismatchEventMED=k_r, ReplacedTotal = ReplacedTotal, KeptTotal = KeptTotal, Total = Total, propReplaced = propReplaced)

        if k_r in pythonH5_dict_all.keys():
            S = dict()
            S['storename'] = np.str(k_r)
            S['sampling_rate'] = float32(0)
            S['timestamps'] = np.array(pythonH5_dict_all[k_r],dtype=float64)
            S['channels'] = np.ones(len(pythonH5_dict_all[k_r]), dtype=int32)
            S['data'] = np.array(range(1,len(pythonH5_dict_all[k_r])+1), dtype=float64)
            S['npoints'] = int(1)
            if os.path.exists(os.path.join(op, k_r +'.hdf5')):
                if not os.path.exists(os.path.join(op, k_r + '_preMEDmatch' +'.hdf5')):
                    os.rename(os.path.join(op, k_r +'.hdf5'), os.path.join(op, k_r + '_preMEDmatch' +'.hdf5'))
                else:
                    os.remove(os.path.join(op, k_r +'.hdf5'))
            save_dict_to_hdf5(S, k_r, op)

# function to read data from 'tsq' and 'tev' files
def readRawData(inputParameters):
    print('### Reading raw data... ###')

    # get input parameters
    inputParameters = inputParameters

    #storesListPath = glob.glob(os.path.join('/Users/VENUS/Downloads/Ashley/', '*_output_*'))

    folderNames = inputParameters['folderNames']
    numProcesses = inputParameters['numberOfCores']
    if numProcesses==0:
        numProcesses = mp.cpu_count()
    elif numProcesses>mp.cpu_count():
        print('Warning : # of cores parameter set is greater than the cores available \
               available in your machine')
        numProcesses = mp.cpu_count()-1

    for i in folderNames:
        filepath = i
        print(filepath)
        storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
        # reading tsq file
        data, flag = readtsq(filepath)
        # checking if doric file exists
        if flag=='tsq':
            pass
        else:
            flag = check_doric(filepath)

        # read data corresponding to each storename selected by user while saving the storeslist file
        for j in range(len(storesListPath)):
            op = storesListPath[j]
            if os.path.exists(os.path.join(op, '.cache_storesList.csv')):
                storesList = np.genfromtxt(os.path.join(op, '.cache_storesList.csv'), dtype='str', delimiter=',')
            else:
                storesList = np.genfromtxt(os.path.join(op, 'storesList.csv'), dtype='str', delimiter=',')

            if isinstance(data, pd.DataFrame) and flag=='tsq':
                execute_readtev(data, filepath, np.unique(storesList[0,:]), op, numProcesses)
            elif flag=='doric_csv':
                execute_import_doric(filepath, storesList, flag, op)
            elif flag=='doric_doric':
                execute_import_doric(filepath, storesList, flag, op)
            else:
                execute_import_csv(filepath, np.unique(storesList[0,:]), op, numProcesses)
            CheckMED(inputParameters, op, filepath, data, storesList)
    print("Raw data fetched and saved.")


if __name__ == "__main__":
    print('run')
    readRawData(json.loads(sys.argv[1]))

