"""
The purpose of this project is to examine data on diabetes patients in the US
and develop code that determines whether a patient is likely to be readmitted 
or not. The data for the project can be found at:
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
along with another csv file explaining the meanings of the various id columns.
The explanation of the different fields can be found at:
https://www.hindawi.com/journals/bmri/2014/781670/tab1/

The code is structured as a series of functions all executed at the end of the
code. It may be worthwhile looking down there first to see in what order things 
executed and to then read the functions (though the functions are defined in 
order of use). Outputs (graphs, printed text) are normally saved in the folder
this program is located in due to the number of them and the occassional 
difficulty in viewing them. Typically there will be a number of graphs (normally)
named after columns and a txt file communicating more numerical information.

A large amount of the data dealt with categorical data, to clarify how it's 
talked about I will be calling the overall information a 'field' which has 
different 'categories'. For example age is a field with the categories 10-20, 
20-30,etc. In most circumstances a field will be interchangable with a column
until the data is one hot encoded, at which point a field will span as many 
columns as is has categories. 

Typical running time is around 3-5 minutes without optimising parameters. 

"""
import sys, os 
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import winsound as ws
import time


from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics as met
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM 

 

def diag_mapper_icd9(diag):
    """Used on the columns diag_1, diag_2 and diag_3.
    Takes a list of icd9 codes and maps them to each category defined by icd9.
    This makes it easier to process in the machine learning algorithms. 
    
    For a list of what each category represents see:
        https://en.wikipedia.org/wiki/List_of_ICD-9_codes
        
    Even reducing down the the number of codes to 19 categories still means
    there's a lot of columns to deal with. For this reason a second mode of this 
    function was created which maps to 5 categories, the 4 most common 
    of the 19 categories and a combined one for everything else. Which of these
    maps is used is determined by the variable 'over'.
    
    Lists can have their values changed by the map method. Map takes a 
    dictionary of values with the starting values as keys and the end values
    as dictionary values. In each part of the if statement the code tries to 
    convert the categories of diag (the input column) to a float. It then looks 
    through the list of categories to map to, checking whether it's between 
    the current element of the list it's looking at and the next one. If it 
    can't make the value into a float it checks to see if that value begins
    with E or V. 
    """

    over = 0
    diag_map = {}
    c_list = diag.unique()
    if over == 0:
        icd9 = [1,140,240,280,290,320,360,390,460,520,580,630,680,710,740,760,
                780,800,1000]
    
        for i in range(len(c_list)):
            try:
                b = float(c_list[i])
                for j in range(len(icd9)-1):
                    if b >= icd9[j] and b < icd9[j+1]:
                        diag_map[c_list[i]] = str(icd9[j])
                        break
                
            except ValueError:
                if c_list[i][0] == 'E':
                    diag_map[c_list[i]] = 'E'
                elif c_list[i][0] == 'V':
                    diag_map[c_list[i]] = 'V'
                
    elif over == 1:
        icd_cut = [[390,460], [240,280], [460,520], [580,630]]
        for i in range(len(c_list)):
            num = 0
            try:
                b = float(c_list[i])
                for j in icd_cut:
                    if b >= j[0] and b < j[1]:
                        diag_map[c_list[i]] = str(j[0])
                        num = 1
                        break
                    
                    if num == 0:
                        diag_map[c_list[i]] = 'Other'
                
            except ValueError:
                diag_map[c_list[i]] = 'Other'
                
                
    diag_new=diag.map(diag_map)
    return diag_new


    

    
    

def data_gen():
    """The purpose of data_gen is to import the data used in this project
and do a basic exploration of it so we understand what the data is like.
Then it will be processed to remove excess information and NaNs as well as to 
condense information into forms easier for machine learning algorithms to process. 
    """
    #get the location of the script and read diabetic_data into memory as a 
    #pandas dataframe
    path = os.path.dirname(sys.argv[0])  +"\\diabetic_data.csv" 
    data = pd.read_csv(path, na_values="?", low_memory=False,index_col=0)
    
    #make a text file called initial data and write information about the data
    #to it.
    print('Output found in '+os.path.dirname(sys.argv[0]))
    size = data.shape
    write_text1 = """Initial data shape is %d rows with %d columns.
This should confirm the website's account of it (101,766 rows, 50 columns, 
minus one column to be used as the index).\n\nNext the head of the data:
"""%(size[0],size[1])
    
    write_text2 = """\nWe now show the categories in each field along with their 
frequencies and the number of NaNs in a field. patient_nbr is discluded
from this since it has %d unique values, making the document difficult to 
read.\n\n""" %(len(data['patient_nbr'].unique()))
    
    outcome =     """
For convinience findings are printed here with supporting data below.     

The majority of columns are categoric, with the exception of a few numerical 
columns outlined in num_list. These are often left as numbers whereas anything 
else is a string.

Notable columns:
diag_1/2/3: can appear to be int, but values are all strings, some have decimals
and V and E occassioanlly come before numbers. E.g. 'E51', 'V25', '250.53'
Needs to be binned into smaller categories.

Weight/payer_code/medical_specialty: all have a lot of NaN values. Probably
worth removing.

patient_nbr: has so many unique values that it was removed from printing. That 
said, only about 70% ofthe patient numbers are unique, so we could remap this
column into the number of times they've been in hospital. 

examide/citoglipton: both only say 'No', should be removed.

readmitted: 3 values, ['NO' '>30' '<30'], only noted for it's importance in the 
project. >/< refers to if they were readmitted before or after 30 days.

Many columns are dominated by a single category, to the point where all other 
categories in the field combined wouldn't make up 1% of the data. These columns
will be dropped since they are unikely to improve the accuracy of the machine
learning algorithms."""
    
    fi = open('initial data'+'.txt','w+')
    fi.write(write_text1)
    fi.write(str(data.head()))
    fi.write(outcome)
    
    fi.write(write_text2)
    for n in data.columns.values[1:]:
        d = data[n].value_counts()
        fi.write(str(d)+'\nNumber of NaNs: ' + str(data[n].isnull().sum())+'\n\n')
    
    fi.close
    
    
    


    #remap patient_nbr to reflect the number of times visited hospital
    #make a dictionary where the keys are patient numbers and the values are
    #the number of times that number is found in the data set.
    #This lets us use this data, in its raw form patient_nbr is unusable.
    nbr = dict(data['patient_nbr'].value_counts())
    #use the dictionary to map numbers to frequency
    a = data['patient_nbr'].map(nbr,na_action=0)
    #call this visit_num and join it to the data set
    visit_num= pd.DataFrame(a.rename("visit_num"))
    data = data.join(visit_num)
    

    """Get rid of the excess columns. 
    Weight, payer code and medical specialty are missing too much data to be 
    useful, trying to include them will just lower the accuracy of the 
    machine learning algorithms. 
    Examide and citoglipton only have 1 category so don't communicate any new 
    information.
    All other fields have such high frequency in one category that 
    duplicating rows would just introduce more biases. The threshold for this is 
    less than 100 samples that aren't in the dominant category. Samples can be 
    duplicated to increase the frequency of a category, but the code implementing
    this will be togglable so that it can be turned off if it doesn't improve
    the program. """
    
    drop_list = ["examide","citoglipton",'patient_nbr','weight','payer_code',
                 'medical_specialty', 'metformin-rosiglitazone', 
                 'acetohexamide','troglitazone','glimepiride-pioglitazone',
                 'metformin-pioglitazone','miglitol','tolazamide',
                 'tolbutamide','glipizide-metformin','chlorpropamide']
    
    #reset the index for ease of use (when imported it uses encounter number)
    #get rid of the columns outlined in drop_list and fill in any nans with the
    #modal value of that column. 
    data = data.reset_index(drop=True)
    data = data.drop(drop_list,axis=1)
    data = data.fillna(data.mode().iloc[0])
    
    #reduce down the possible values in the diag columns using the function
    #diag_mapper_icd9, defined above. This will allow one hot encoding to be
    #used later on.
    data['diag_1']=diag_mapper_icd9(data['diag_1'])
    data['diag_2']=diag_mapper_icd9(data['diag_2'])
    data['diag_3']=diag_mapper_icd9(data['diag_3'])
    
    data['readmitted']=data['readmitted'].map({'NO':1,'<30':0,'>30':0})
    
#map the id data to group similar fields. Some of these are low frequency 
#some just communicate the same information e.g. anything mapped to s_null is
#a category where there's no information on what the field means.
#This doesn't improve accuracy but does reduce the running time. 
#see IDs_mapping.csv for examples
    tostr = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    #enusre that the data in these columns are integers
    data[tostr] = data[tostr].astype(type(3))
    
    #construct dictionaries for the mapping
    o = 'other'
    e = 'dead'
    h = 'hospice'
    c = 'child_newborn'
    dh = 'discharged to health facility'
    transfer = 'transfer'
    s_null = 'lim info'
    ref = 'other referral'
    d_map = {1:'Home',
             2:dh,#only 1 3 and 6 don't need to be mapped, though could be anyway
             3:'Dis to SNF',
             4:dh,
             5:dh,
             6:'Home w Home Care',
             7:'left AMA',
             8:dh,
             9:'inpatient',
             10:dh,
             11:e,
             12:'pat_outpat',
             13:h,
             14:h,
             15:dh,
             16:dh,
             17:dh,
             18:s_null,
             19:e,
             20:e,
             21:e,
             22:dh,
             23:dh,
             24:dh,
             25:s_null,
             26:s_null,
             27:dh,
             28:dh,
             29:dh,
             30:dh}
             
    
    s_map = {1:'Physician Referral',
             2:ref,#1 7 and 17 don't have to be mapped
             3:ref,
             4:transfer,
             5:transfer,
             6:transfer,
             7:'Emergency room',
             8:o,
             9:s_null,
             10:transfer,
             11:o,
             12:o,
             13:c,
             14:o,
             15:s_null,
             17:s_null,
             18:transfer,
             19:o,
             20:s_null,
             21:s_null,
             22:transfer,
             23:c,
             24:c,
             25:transfer,
             26:transfer
             }
    t_map = {1:'Emergency',
             2:'Urgent',
             3:'Elective',
             4:o,
             5:s_null,
             6:s_null,
             7:o,
             8:s_null}
             
             
    
    data['admission_source_id']=data['admission_source_id'].map(s_map)
    data['admission_type_id']=data['admission_type_id'].map(t_map)
    data['discharge_disposition_id']=data['discharge_disposition_id'].map(d_map)
              
    return data

def exploratory(data,folder_name):
    """Generates graphs of data and saves them in a folder in the current
    working directory called folder_name for further viewing. Returns to the 
    starting directory when finished. """
    current_path = os.path.dirname(sys.argv[0]) 
    path = current_path +"\\"+folder_name
    print('New images in '+folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    
    fi = open(folder_name+'.txt','w+')
    fi.write('Frequency of each category found in the data set.\n')
    bar_list = data.columns.values
    perc_readmit = float(data['readmitted'].sum())/float(data.shape[0])
    for n in bar_list:
        d = data[n].value_counts()
        
        f, ax = plt.subplots(figsize=(len(d)*3+10, 12))
        ax = sns.barplot(x=n,y='readmitted', data=data)
        plt.plot(np.linspace(-0.5,len(data[n].unique())-0.5,1000), 
                 [perc_readmit]*1000, 'r')
         
        fi.write(str(d)+'\n\n')
        fig = ax.get_figure()
        fig.savefig(n+'.png')
        plt.close()
        plt.close()
     
    fi.close
    os.chdir(current_path)
    

def filter_data(adata,filt,n_std):
    """This function focuses on filtering out outliers and altering low
    frequency categories. It should only be applied to the training data. 
    
    adata is the data to filter.
    filt controls whether to filter out outliers on numeric data.
    n_std determines the number of standard deviations beond which marks an 
    outlier.
    """
    
    
    
    data,test = train_test_split(adata,test_size=0.2)
    if not drop:
        print('Not dropping/padding low freq categories')
        
    #Dealing with Low Frequency Data  
    #If a category has less than dump_tresh values, dump it.
    #If it has more than dump_thresh but less than pad_thresh values, duplicate
    #It so there are roughly pad_to values so it's more prominant.
    dump_thresh = round(1*data.shape[0]/100)
    pad_thresh = 2*dump_thresh
    pad_to = pad_thresh*10#round(data.shape[0]/3)
    
    """For every column in the list of categorical columns, cat_list, get a 
    series, low, of all of the categories in a field with frequencies less than 
    dump_thresh and replace them with the mode.
    
    If drop, drop any columns which now have 1 or less categories in them and 
    remove that column name from the relevent lists. Duplicate rows which 
    are within the padding range discussed earlier."""
    for i in cat_list:
        desc = data[i].describe()
        freq = data[i].value_counts()
        low = freq[freq <dump_thresh]
        
        if len(low)>0:
            #
            data.loc[:,i] = data[i].replace(low.index.tolist(), desc[2])
            test.loc[:,i] = test[i].replace(low.index.tolist(), desc[2])
        if drop:
            freq = data[i].value_counts()
            if len(freq) <= 1:
                
                data = data.drop(i, axis=1)
                cat_list.remove(i)
                aless_cat_list.remove(i)
                med_list.remove(i)
                uneven_list.remove(i)
                
            else:
                
                
                if not pad_low:
                    pad =  freq[freq <pad_thresh]
                    pad_list = pad.index.tolist()
                    
                    for j in pad_list:
                        f = int(math.ceil(pad_to/freq[j]))
                        data = data.append([data[data[i]==j]]*f,ignore_index=True)
                        

    print('Low frequency checks done')
    if drop_dup:
        data=data.drop_duplicates()
        print('Dropping duplicates')
   
    
    #filter out any numeric information more or less than n_std times the 
    #standard deviation to reduce noise from outliers.
    if filt:
        desc_0 = data.describe()
        for i in list(desc_0.columns.values):
            data = data[(data[i] < desc_0[i][1]+n_std*desc_0[i][2])] 
            data = data[(data[i] > desc_0[i][1]-n_std*desc_0[i][2])]  
        

    return data,test




def dummy_data(data,lab_list):
    """Process data so it is fit for machine learning algorithms. This focuses
    on encoding categoric data.
    
    data is the data to be processed, lab_list is a list of columns to be label
    encoded."""

    #nlab_list is the list of the remaining categoric columns which will be 
    #one hot encoded
    nlab_list = list(set(aless_cat_list)-set(lab_list))
    
    for i in lab_list:
        le = LabelEncoder()
        le.fit(list(data[i].unique()))
        data.loc[:,i] = le.transform(data[i])
    prepped_data = data
    prepped_data = pd.get_dummies(data, prefix=nlab_list, prefix_sep='_',  
                                  columns=nlab_list, drop_first=True)
    
    pans = prepped_data.readmitted
    pdata = prepped_data.drop(['readmitted'],axis=1)
    
    if scale:
        s = StandardScaler()
        temp = s.fit_transform(pdata)
        pdata = pd.DataFrame(temp,columns=pdata.columns.values,index = pdata.index)
        
    return  prepped_data,pdata,pans



def scaled_graphs(pdata,pans, graphs,heatmap):
    """Exploratory heatmap of the principle components. 
    
    graphs is a boolean that controls whether to generate graphs of every 
    present feature wih colour decided by readmitted"""
    pca = PCA()
    ind = ['PC'+str(i+1) for i in range(pdata.shape[1])]  #'PC' is just a collumn name used during scores
    #ind = ['PC'+str(i+1) for i in range(ncomp)]
    # Create the PCA scores matrix and check the dimensionality of the PCA scores
    scores = pca.fit_transform(pdata)
    scores = pd.DataFrame(scores, columns = ind, index = pdata.index)
    
    if heatmap:
        loadings = pca.components_
        loadings = pd.DataFrame(loadings, columns = pdata.columns, index = ind)
        
        f, ax = plt.subplots(figsize=(60, 32))
        ax = sns.heatmap(loadings.transpose(), linewidths=0.5, cmap="RdBu", 
                         vmin=(-1), vmax=1, annot=False)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
        ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, 
                           fontsize=12)
        fig = ax.get_figure()
        fig.savefig('heatmap.png')
        plt.close()
        
    current_path = os.path.dirname(sys.argv[0]) 
    path = current_path +"\\graphs vs readmitted"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    
    if graphs:
        col_list = list(pdata.columns.values)  
        for i in col_list:
            ax = sns.countplot(pdata[i], hue=pans)
            #plt.show()
            fig = ax.get_figure()
            fig.savefig(i+'.png')
            plt.close()
    
    
    os.chdir(current_path)

    
def unsup(pdata):
    
    """unsup is where unsupervised learning takes place. It makes use of KMeans,
    a clustering algorithm based on the distance of points from an initially
    randomly chosen cluster centre.
    
    KMeans was initially chosen for its speed and ability to deal with large
    datasets and was later found to be one of sklearn's two algorithms that
    could cope with the size. DBSCAN (the other algorithm) took a long time to 
    run and left large sections of data unclustered. KMeans' only required 
    input is the number of clusters to look for. I chose 30, which gives a 
    reasonable sillouhette without taking too long to run. Even so, due
    to the categorical nature of most of the data clusters will often fit to
    one category and not represent a more rounded group of patients. A better
    algorithm would be KModes, which instead of using the mean of points to 
    determine the cluster centre uses the mode, however sklearn doesn't support
    this algorithm.""" 
    
    
    data = pdata
    pdata = pd.DataFrame(StandardScaler().fit_transform(pdata))
    print('Starting clustering')
    scan_data = KMeans(n_clusters=30).fit(pdata.iloc[:,:])
    cluster_assignment = scan_data.fit_predict(pdata.iloc[:,:])
    
    print('Clustering done')
    
    
    c_data = data.join(pd.Series(cluster_assignment, index=data.index, 
                                 name='cluster_assignment'), how='left')
    c_data = c_data.reset_index(drop=True)
    
    
    current_path = os.path.dirname(sys.argv[0]) 
    path = current_path +"\\clustering graphs"
    print('New images in folder clustering graphs')
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    
    for n in c_data.columns.values:
        d = c_data[n].value_counts()
        
        f, ax = plt.subplots(figsize=(len(d)*3+10, 12))
        ax = sns.countplot(c_data['cluster_assignment'],hue=c_data[n])
        fig = ax.get_figure()
        name = str(n).replace('>','')
        name = name.replace('<','')
        fig.savefig(name+'.png')
        plt.close
        
    os.chdir(current_path)

    
    
def param(data,ans):
    
    """param's purpose is to find the optimal parameters for the supervised 
    learning algorithm used.
    Takes the training data and the answers to readmittance as separate 
    arguments. Returns a dictionary of parameters for use in the the full 
    algorithm.
    """
    print('Starting supervised optimization')
    #make lists of parameters to test. hidden_layer_sizes need to be tuples.
    alp = 10.0 **-np.arange(1, 4)
    hid2 = np.arange(60,120,10) 
    hid = []
    for i in range(len(hid2)):
        hid.extend((hid2[i],))
    #turn into a dictionary of parameters
    parameters = [{'alpha': alp, 'hidden_layer_sizes':hid}]
    # Optimise and build the model with GridSearchCV
    gridCV = GridSearchCV(MLPClassifier(), parameters, cv=5)
    gridCV.fit(data, ans)     
    
        
    
    # Report the optimal parameters
    final_parameters = gridCV.best_params_    
    for k in final_parameters:
        print('Best '+k+' = '+str(final_parameters[k]))
        
    final_parameters['hidden_layer_sizes'] = (final_parameters['hidden_layer_sizes'],)
        
    return final_parameters
    
def sup_learning(train_data,train_ans,test_data,test_ans,param):
    """sup_learning does the proper supervised learning and clasification of
    the test data, then prints the confusion matrix and several metrics.
    
    The algorithm used is from sklearn's neural networking module and uses the 
    Multilayer Perceptron technique. MLP was the fastest and most accurate of 
    all the algorithms tried, giving accuracies of around 80%.
    """
    clf = MLPClassifier()
    clf.set_params(**param)
    clf.fit(train_data, train_ans)
    ans = clf.predict(test_data)
        
    
    tn, fp, fn, tp = confusion_matrix(test_ans, ans ).ravel()
    print('True negatives: ',tn,'\nFalse positives: ',fp,'\nFalse negatives: ',fn,'\nTrue positives: ',tp)
    print ("Overall Accuracy:", round(met.accuracy_score(test_ans, ans), 2))
    print ("Overall Precision:", round(met.precision_score(test_ans, ans), 2))
    print ("Overall Recall:", round(met.recall_score(test_ans, ans), 2))
    astr = """\tAccuracy represents ability to classify points correctly. [(tp+tn)/all]
\tPrecision represents ability to correctly identify positive data. [tp/(tp + fp)]
\tRecall represents the ability to find all the positve data. [tp/(tp + fn)]"""
    print(astr)
    return ans
    


#----------------------------------------------------------------
#functions have been defined. Program can now be executed.

start_time = time.time()



global num_list
global cat_list
global aless_cat_list
global uneven_list
global med_list

#list of column names to be used in multiple functions
num_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', 'number_diagnoses']

cat_list = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
            'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 
            'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 
            'nateglinide', 'glimepiride', 
            'glipizide', 'glyburide',  'pioglitazone', 
            'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin', 
            'change', 'diabetesMed', 'readmitted']
                 
aless_cat_list = ['race', 'gender',  'admission_type_id', 'discharge_disposition_id', 
                  'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 
                  'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 
                  'nateglinide', 'glimepiride', 
                  'glipizide', 'glyburide',  'pioglitazone', 
                  'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin', 
                  'change', 'diabetesMed']
#uneven_list is a list of all collumns which have a low freq category         
uneven_list = ['diag_3', 'diag_1','diag_2','rosiglitazone', 'gender', 'pioglitazone',
               'admission_source_id', 'repaglinide', 'discharge_disposition_id',
               'acarbose','glyburide-metformin', 'admission_type_id', 'nateglinide']

med_list = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride', 
            'glipizide', 'glyburide',  'pioglitazone', 'rosiglitazone', 
            'acarbose', 'insulin', 'glyburide-metformin']

global scale
global drop
global pad_low
global drop_dup

#booleans used in several funcions to toggle certain effects
scale = False#standard scale the data used?
drop = False#should low frequency data be dropped
pad_low = True#pad low freq cat that don't get droppeddrop_dup = False#drop all duplicate rows 
drop_dup = False#drop duplicate rows. 

#Supress graph outputs to the console 
plt.ioff()

#Import the data, output some information about its basic shape,
#drop collumns that aren't useful, alter columns where there are too
#many values.
full_data= data_gen()

#Save a series of graphs looking at the shape of the data in a folder
#called 'initial graphs' for user study.
exploratory(full_data,'initial graphs')

#Filter the data. This may involve dropping poorly populated categories in 
#fields or replacing them with the mode. Also replacing numerical outliers
#with the mode.
train,test = filter_data(full_data,1,3)

#Save a series of graphs looking at the the data once it is filtered in a folder
#called 'post filtering graphs'.
exploratory(train,'post filtering graphs')

#Drop any columns in test that got removed from train during filtering
test = test.drop(list(set(test)-set(train)),axis=1)

#lab_list is a list of columns to be label encoded
#age is always label encoded because it is ordinal
labels = 0
if labels == 1:
    lab_list = ['age','diag_1','diag_2','diag_3','admission_type_id', 
            'discharge_disposition_id', 'admission_source_id'] 
elif labels == 2:
    lab_list = aless_cat_list  +['age']     
else:
    lab_list = ['age']     
    
#Process the data for learning algorithms. Some values will be label encoded,
#the remainder are one hot encoded. 
train_full,train_data,train_ans = dummy_data(train,lab_list)
test_full,test_data,test_ans = dummy_data(test,lab_list)


scaled_graphs(train_data,train_ans,False,True)#pdata,pans, graphs,heatmap

"""Challenges:
    @dealing with diag
        The first real challenge I had on this project was figuring out how to 
        manage the diag columns. Each had over 800 values which seemed impossible
        to one hot or label encode. It wasn't until I found the webpage describing
        what each field meant that I was able to deal with it. what seemed like
        random number now had a structure to them, and I could sort the data 
        into bins based on that. After that it was simple enough to learn about
        mapping and so diag_mapper was created. 
        Key Points:  Understand what the data means and how it's presented. 
                     Basic research can lead to better design.
    @good design
        Initially this program was written in a jupyter notebook, this was a 
        good tool to begin with, when much of the work relied on outputting 
        information and rerunning small chunks of code. But as the code got 
        longer it became more difficult to edit code and keep track of variables,
        So it was transferred to Spyder and divided up into functions. The 
        functions made it easier to debug and run but the code would be more 
        elegant if I'd written it with the mind that it would be in spyder
        eventually. Alternatively there's nothing stopping me defining 
        functions in jupyter, though this doesn't make as good a use of its 
        format.
        Key Points: Consider the final size of your code when starting out,
                    design with that in mind.  
    @patient_nbr
        When I first started working on this project I tried printing out all 
        unique values of every column. This was a mistake. Patient_nbr has 
        70,000~ unique values, that data was unworkable so I dropped it. Later
        when I came to use the supervised learning algorithms I was disappointed
        by an accuracy of at most 60%. It took a long time to realise that I 
        could turn id numbers into frequency of visits, which then raised the
        the accuracy to 80%.
        Key Points: Don't disregard data until you're convinced it's worthless.
                    Something that looks difficult process may just need 
                    converting to a new form. 
"""

#Run an unsupervised learning algorithm then optimise the supervised learning 
#algorithm.
unsup(train_full)


#Optimise the supervised learning algorithm and return a dictionary of the 
#best parameters to use. param has a long run time so it can be commented out 
#and a default value of par_dict can be used instead.
par_dict = {'alpha': 0.1, 'hidden_layer_sizes':(60,)}
par_dict=param(train_data,train_ans)
 
test_data = test_data.drop(list(set(test_data)-set(train_data)),axis=1)

#run the algorithm, output the confusion matrix and some basic stats on performance
#report how long it took to run and beep when finished
pred_ans = sup_learning(train_data,train_ans,test_data,test_ans,par_dict)

print("--- %s seconds ---" % (time.time() - start_time))
ws.Beep(800,1000)
print("finish sup learning")
