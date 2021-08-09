import pyprind
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import pandas as pd
from string import punctuation
import numpy as np
import math
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
from itertools import chain, combinations
import tensorflow as tf
import matplotlib.pyplot as plt

########################################DATA ENG START#######################################################
def build_trace_compliance(myData):
    trace = myData.MILESTONE_NAME.str.cat(sep=',')
    adjustmentCompliancePercent = pd.to_numeric(myData.adjustmentCompliance, errors='coerce').mean()
    collectionCompliancePercent = round(pd.to_numeric(myData.collectionCompliance, errors='coerce').mean())
    dosingCompliancePercent = round(pd.to_numeric(myData.dosingCompliance, errors='coerce').mean())
    c = ['trace', 'adjustmentCompliancePercent', 'collectionCompliancePercent', 'dosingCompliancePercent']
    return pd.Series([trace, adjustmentCompliancePercent, collectionCompliancePercent, dosingCompliancePercent],index=c)

def build_static_compliance(myData):
    adminAgeYrs=round(pd.to_numeric(myData['ADMIN_AGE_YRS'], errors='coerce').mean())
    adminMedService=myData['ADMIN_MED_SERVICE'].iloc[0]
    adminFacility = myData['ADMIN_FACILITY'].iloc[0]
    adminUnit = myData['ADMIN_UNIT'].iloc[0]
    adjustmentCompliancePercent = round(pd.to_numeric(myData.adjustmentCompliance, errors='coerce').mean())
    collectionCompliancePercent = round(pd.to_numeric(myData.collectionCompliance, errors='coerce').mean())
    dosingCompliancePercent = round(pd.to_numeric(myData.dosingCompliance, errors='coerce').mean())
    c = ['adminAgeYrs','adminMedService','adminFacility','adminUnit','adjustmentCompliancePercent', 'collectionCompliancePercent', 'dosingCompliancePercent']
    return pd.Series([adminAgeYrs,adminMedService,adminFacility,adminUnit,adjustmentCompliancePercent,collectionCompliancePercent, dosingCompliancePercent],index=c)

#DATA MUNGING
#create some values for conversion
nonCompliantAdjustment=["Noncompliant","Noncompliant 30min HOLD","Noncompliant 60min HOLD","Noncompliant 120min HOLD"]
compliantAdjustment=["Compliant Adjustment Time","Compliant 30min HOLD","Compliant 60min HOLD","Compliant 120min HOLD"]
compliantCollection=["Compliant Collection Time"]
nonCompliantCollection=["Early/Late Collection Time","No Lab Collected w/in 12 hrs"]
nonCompliantDosing=["Noncompliant"]
compliantDosing=["Compliant Dose","Compliant Supratherapeutic Turnoff"]
#myComplianceMHGR.dropna(inplace=True)

myMilestoneData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_hep_milestone_110117_110118_009.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},error_bad_lines=False,sep='\t',encoding = "ISO-8859-1").drop_duplicates()
myMilestoneData[['TIMESTAMP']]=myMilestoneData[['TIMESTAMP']].apply(pd.to_datetime,format='%d-%b-%Y %H:%M:%S',errors='ignore')
myMilestoneData.ENCNTR_ID= myMilestoneData.ENCNTR_ID.str.split(".",expand=True)[0]
myMilestoneData.ADMIN_EVENT_ID= myMilestoneData.ADMIN_EVENT_ID.str.split(".",expand=True)[0]
myMilestoneData=myMilestoneData[-myMilestoneData.MILESTONE_NAME.str.contains("ptt|protamine_admin|heparin_admin_charted|heparin_begin_bag")]
myDemographicData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_hep_protocol_110117_110118_009.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},sep='\t',encoding = "ISO-8859-1")

#subset for what we need
myDemographicData=myDemographicData[["CLIENT","ADMIN_AGE_YRS","LANGUAGE","MARITAL_STATUS","RACE","RELIGION","ENCNTR_ID","REG_PRSNL_UN","REG_MED_SERVICE","REG_FACILITY","REG_UNIT","DISCH_DISPOSITION","PATHWAY_CATALOG_ID","ORDER_SET","ORDER_MNEMONIC","CATALOG_CD","ORDER_MED_SERVICE","ORDER_FACILITY","ORDER_UNIT","ORDER_SENTENCE","ORDER_WEIGHT","ORDER_WEIGHT_UNITS","ORDERING_PROVIDER_UN","ORDERING_PROVIDER_POSITION","ADMIN_PRSNL_ID","ADMIN_PRSNL_UN","ADMIN_PRSNL_POSITION","ADMIN_MED_SERVICE","ADMIN_FACILITY","ADMIN_UNIT","ADMIN_UNIT_DESCRIPTION","ADMIN_ENCNTR_TYPE","ADMIN_CATALOG_CD","ADMIN_EVENT_CD","ADMIN_EVENT_CD_DISP","ADMIN_EVENT_ID","ADMIN_PARENT_EVENT_ID","ADMIN_INFUSION_RATE","ADMIN_INFUSION_UNITS","ADMIN_RESULT_STATUS","ADMIN_EVENT_TAG","ADMIN_VERIFIED_PRSNL_ID","ADMIN_WITNESS_BY_UN","CHARTING_APP","PT_SCANNED_IND","MED_SCANNED_IND","PUMP_PROGRAMMING_IND","PUMP_PROG_ELIGIBLE_IND","RX_ROUTE","PENDING_VERIFY_AT_ADMIN_IND","VERIFY_STATUS_AT_ADMIN","DRC_ALERT_CNT","DRC_ACCEPT_CNT","DRUGDRUG_ALERT_CNT","DRUGDRUG_ACCEPT_CNT","DRUGDUP_ALERT_CNT","DRUGDUP_ACCEPT_CNT","DRUGALLERGY_ALERT_CNT","DRUGALLERGY_ACCEPT_CNT","OTHER_DISCERN_ALERT_CNT","OTHER_DISCERN_ACCEPT_CNT","REV_ADMIN_IND","PTT_COLL_RESULT_STATUS","PTT_COLL_RESULT_VAL","PTT_COLL_RESULT_UNITS","PTT_COLL_UNIQUE_TO_ADMIN","PTT_COLL_ORDER_DT_TM","PTT_COLL_OD_REQ_START_DT_TM","PTT_COLL_OD_PRIORITY","PTT_COLL_OD_SPECIMEN_TYPE","PTT_COLL_OD_NURSECOLLECT","PTT_COLL_OA_COLLECTED","PTT_COLLECTED_BY_UN","PTT_COLLECTED_BY_POSITION","PTT_COMP_RESULT_STATUS","PTT_COMP_RESULT_VAL","PTT_COMP_RESULT_UNITS","DX_LIST","DX_PRIORITY_LIST","DX_TYPE_LIST"]]
myDemographicData.ENCNTR_ID= myDemographicData.ENCNTR_ID.str.split(".",expand=True)[0]
myDemographicData.ADMIN_EVENT_ID= myDemographicData.ADMIN_EVENT_ID.str.split(".",expand=True)[0]
myComplianceMHGR = pd.read_csv("C:/Users/an052283/Desktop/MHGR_DC V29 Compliance by Admin (w Mid).csv",dtype={'Admin Event Id':str}).drop_duplicates()
myComplianceMHGR.rename(index=str, columns={"Admin Event Id": "ADMIN_EVENT_ID","Adjustment Time Compliance": "ADJUSTMENT_TIME_COMPLIANCE", "Dosing Compliance":"DOSING_COMPLIANCE","Collection Time Compliance":"COLLECTION_TIME_COMPLIANCE"},inplace=True)

#compliant is 0, non-compliant is 1
myComplianceMHGR['adjustmentCompliance'] = pd.NaT
myComplianceMHGR.loc[myComplianceMHGR['ADJUSTMENT_TIME_COMPLIANCE'].isin(compliantAdjustment),'adjustmentCompliance'] = 0
myComplianceMHGR.loc[myComplianceMHGR['ADJUSTMENT_TIME_COMPLIANCE'].isin(nonCompliantAdjustment),'adjustmentCompliance'] = 1
myComplianceMHGR['collectionCompliance'] = pd.NaT
myComplianceMHGR.loc[myComplianceMHGR['COLLECTION_TIME_COMPLIANCE'].isin(compliantCollection),'collectionCompliance'] = 0
myComplianceMHGR.loc[myComplianceMHGR['COLLECTION_TIME_COMPLIANCE'].isin(nonCompliantCollection),'collectionCompliance'] = 1
myComplianceMHGR['dosingCompliance'] = pd.NaT
myComplianceMHGR.loc[myComplianceMHGR['DOSING_COMPLIANCE'].isin(compliantDosing),'dosingCompliance'] = 0
myComplianceMHGR.loc[myComplianceMHGR['DOSING_COMPLIANCE'].isin(nonCompliantDosing),'dosingCompliance'] = 1

sJKeys = ["ADMIN_EVENT_ID"]
myMilestoneData=pd.merge(myMilestoneData,myComplianceMHGR,how='inner',on=sJKeys)
sJKeys = ["CLIENT","ENCNTR_ID","ADMIN_EVENT_ID"]
myMilestoneData=pd.merge(myMilestoneData,myDemographicData,how='inner',on=sJKeys)
myMilestoneData.dropna(subset=["collectionCompliance","adjustmentCompliance","dosingCompliance"], how='any',inplace=True)
myMilestoneData.sort_values(by="TIMESTAMP",inplace=True)

traceCompliances=myMilestoneData.groupby(['CLIENT','ENCNTR_ID','PATHWAY_CATALOG_ID']).apply(build_trace_compliance).reset_index()
#traceCompliances.to_csv('C:/Users/an052283/Desktop/MHGR_mo_hep_trace_compliances.tsv',sep='\t',doublequote=True,index=True)

#milestone names have underscore "_" in names - should probably use camel notation in future but for time's sake use this workaround for now
punctuationNoUnderscore= punctuation.replace("_","")

counts = Counter()

pbar = pyprind.ProgBar(len(traceCompliances['trace']),
                       title='Counting words occurences')
for i,trace in enumerate(traceCompliances['trace']):
    text = ''.join([c if c not in punctuationNoUnderscore else ' '+c+' ' \
                    for c in trace]).lower()
    traceCompliances.loc[i,'trace'] = text
    pbar.update()
    counts.update(text.split())

## Create a mapping:
## Map each unique word to an integer

word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}

mapped_traces = []
pbar = pyprind.ProgBar(len(traceCompliances['trace']),
                       title='Map traces to ints')
for trace in traceCompliances['trace']:
    mapped_traces.append([word_to_int[word] for word in trace.split()])
    pbar.update()

## Define fixed-length sequences:
## Use the last 200 elements of each sequence
## if sequence length < 200: left-pad with zeros
sequence_length = traceCompliances.trace.map(lambda x: x.count(' ')).max() +2  ## sequence length (or T in our formulas)
sequences = np.zeros((len(mapped_traces), sequence_length), dtype=int)
for i, row in enumerate(mapped_traces): #i represents the first dimension in the two dimensional numpy ndarray (here nd means 'n' dimensional), i:j:k equals starting index, stopping index and steps.  Steps can be backward if j is negative and can go high to low if k is negative
    trace_arr = np.array(row)
    sequences[i, -len(row):] = trace_arr[-sequence_length:]  #sequence length versus len(row) ?

#TRACE DATA
x_sequence_train = sequences[:round(traceCompliances.shape[0]/2)]
y_sequence_train = traceCompliances.dosingCompliancePercent[:round(traceCompliances.shape[0]/2)].values
x_sequence_test = sequences[round(traceCompliances.shape[0]/2):]
y_sequence_test = traceCompliances.dosingCompliancePercent[round(traceCompliances.shape[0]/2):].values

#STATIC DATA
static_trace_data =myMilestoneData.groupby(['CLIENT','ENCNTR_ID','PATHWAY_CATALOG_ID']).apply(build_static_compliance).reset_index()
static_trace_data=pd.get_dummies(static_trace_data[['adminMedService','adminFacility','adminUnit']])
x_static_train = static_trace_data[:round(static_trace_data.shape[0]/2)]
x_static_test = static_trace_data[round(static_trace_data.shape[0]/2):]

#LOGIC

np.random.seed(123) # for reproducibility

# Step 1: Defining multilayer RNN cells
# Step 2: Defining the initial states for the RNN cells
# Step 3: Creating the recurrent neural network using the RNN cells and their states

## Train:

n_words = max(list(word_to_int.values())) + 1

# Headline input: meant to receive sequences of "sequence_length" integers, between 1 and n_words.
# Note that we can name any layer by passing it a "name" argument.
sequence_input = Input(shape=(sequence_length,), dtype='int32', name='sequence_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=sequence_length)(sequence_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

#Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly
#Even though the main loss will be much higher in the model
static_output = Dense(1, activation='sigmoid', name='static_output')(lstm_out)

#At this point, we feed into the model our auxiliary input data by concatenating it with the LSTM output:
auxiliary_input = Input(shape=(x_static_train.shape[1],), name='static_input')

x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#And finally we add the main logistic regression layer
sequence_output = Dense(1, activation='sigmoid', name='sequence_output')(x)
#This defines a model with two inputs and two outputs:
model = Model(inputs=[sequence_input, auxiliary_input], outputs=[sequence_output, static_output])

#Since our inputs and outputs are named (we passed them a "name" argument), we could also have compiled the model via:
model.compile(optimizer='rmsprop',
              loss={'sequence_output': 'binary_crossentropy', 'static_output': 'binary_crossentropy'},
              loss_weights={'sequence_output': 1., 'static_output': 0.2})
# And trained it via:
model.fit({'sequence_input': x_sequence_train[:32], 'static_input': x_static_train[:32]},
          {'sequence_output': y_sequence_train[:32], 'static_output': y_sequence_train[:32]},
          epochs=2, batch_size=32)

#potential throwaway

#static_train[["collectionCompliance","adjustmentCompliance","dosingCompliance"]] = myMilestoneData[["collectionCompliance","adjustmentCompliance","dosingCompliance"]]
#static_train['ADMIN_AGE_YRS']=pd.to_numeric(myMilestoneData['ADMIN_AGE_YRS'],errors='coerce').round()
# myMilestoneData['ADMIN_AGE_YRS'] = pd.to_numeric(myMilestoneData['ADMIN_AGE_YRS'],errors='coerce')
#try and get shapes to match and pass through model tomorrow
# # We normalize the age and the fare by subtracting their mean and dividing by the standard deviation
# age_mean = static_train['ADMIN_AGE_YRS'].mean()
# age_std = static_train['ADMIN_AGE_YRS'].std()
# static_train['ADMIN_AGE_YRS'] = (static_train['ADMIN_AGE_YRS'] - age_mean) / age_std

# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set
#x_static_train = static_train.drop(["dosingCompliance"], axis=1).as_matrix()
#y_static_train = static_train[["dosingCompliance"]].as_matrix()

