#OLD
#The main input will receive the headline, as a sequence of integers (each integer encodes a word).
#The integers will be between 1 and 10,000 (a vocabulary of 10,000 words) and the sequences will be 100 words long.
#NEW
#The main input will receive the headline, as a sequence of integers (each integer encodes a word).
#The integers will be between 1 and n_words (a vocabulary of n_words) and the sequences will be sequence_length words long.
import pyprind
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import pandas as pd
from string import punctuation
import re
import numpy as np
import math
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
from itertools import chain, combinations
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt


def build_trace_compliance(myData):
    trace = myData.MILESTONE_NAME.str.cat(sep=',')
    adjustmentCompliancePercent = pd.to_numeric(myData.adjustmentCompliance, errors='coerce').mean()
    adjustmentCompliancePercent = round(pd.to_numeric(myData.adjustmentCompliance, errors='coerce').mean())
    collectionCompliancePercent = round(pd.to_numeric(myData.collectionCompliance, errors='coerce').mean())
    #collectionCompliancePercent = pd.to_numeric(myData.collectionCompliance, errors='coerce').mean()
    #dosingCompliancePercent = pd.to_numeric(myData.dosingCompliance, errors='coerce').mean()
    dosingCompliancePercent = round(pd.to_numeric(myData.dosingCompliance, errors='coerce').mean())
    # largestHgbDropPercentTimeDelta=pd.to_numeric(x.largestHgbDropPercentTimeDelta,errors='coerce').max()
    # male=pd.to_numeric(np.where(x.SEX == "M",1,0),errors='coerce').max()
    # grbc=pd.to_numeric(x.GRBC,errors='coerce').max()
    # age=pd.to_numeric(x['ADMIN_AGE_YRS'],errors='coerce').mean().round()
    # medianHgb=pd.to_numeric(x[x['eventCodeMapped'] == 'Hgb']['RESULT_VAL'],errors='coerce').median()
    # lowestHgb=pd.to_numeric(x[x['eventCodeMapped'] == 'Hgb']['RESULT_VAL'],errors='coerce').min()
    # c = ['medianHgb','lowestHgb','age','largestHgbDropPercent','largestHgbDropPercentTimeDelta','male','grbc']
    # return pd.Series([medianHgb,lowestHgb,age,largestHgbDropPercent,largestHgbDropPercentTimeDelta,male,grbc],index=c)
    c = ['trace', 'adjustmentCompliancePercent', 'collectionCompliancePercent', 'dosingCompliancePercent']
    return pd.Series([trace, adjustmentCompliancePercent, collectionCompliancePercent, dosingCompliancePercent],index=c)

#initialize for later
counts = Counter()

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
#myMilestoneData = myMilestoneData[(myMilestoneData['CLIENT'] == "MHGR_DC")]
myMilestoneData=myMilestoneData[-myMilestoneData.MILESTONE_NAME.str.contains("ptt|protamine_admin|heparin_admin_charted|heparin_begin_bag")]
myDemographicData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_hep_protocol_110117_110118_009.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},sep='\t',encoding = "ISO-8859-1")
#subset for what we need
myDemographicData=myDemographicData[["CLIENT","ADMIN_AGE_YRS","LANGUAGE","MARITAL_STATUS","RACE","RELIGION","ENCNTR_ID","REG_PRSNL_UN","REG_MED_SERVICE","REG_FACILITY","REG_UNIT","DISCH_DISPOSITION","PATHWAY_CATALOG_ID","ORDER_SET","ORDER_MNEMONIC","CATALOG_CD","ORDER_MED_SERVICE","ORDER_FACILITY","ORDER_UNIT","ORDER_SENTENCE","ORDER_WEIGHT","ORDER_WEIGHT_UNITS","ORDERING_PROVIDER_UN","ORDERING_PROVIDER_POSITION","ADMIN_PRSNL_ID","ADMIN_PRSNL_UN","ADMIN_PRSNL_POSITION","ADMIN_MED_SERVICE","ADMIN_FACILITY","ADMIN_UNIT","ADMIN_UNIT_DESCRIPTION","ADMIN_ENCNTR_TYPE","ADMIN_CATALOG_CD","ADMIN_EVENT_CD","ADMIN_EVENT_CD_DISP","ADMIN_EVENT_ID","ADMIN_PARENT_EVENT_ID","ADMIN_INFUSION_RATE","ADMIN_INFUSION_UNITS","ADMIN_RESULT_STATUS","ADMIN_EVENT_TAG","ADMIN_VERIFIED_PRSNL_ID","ADMIN_WITNESS_BY_UN","CHARTING_APP","PT_SCANNED_IND","MED_SCANNED_IND","PUMP_PROGRAMMING_IND","PUMP_PROG_ELIGIBLE_IND","RX_ROUTE","PENDING_VERIFY_AT_ADMIN_IND","VERIFY_STATUS_AT_ADMIN","DRC_ALERT_CNT","DRC_ACCEPT_CNT","DRUGDRUG_ALERT_CNT","DRUGDRUG_ACCEPT_CNT","DRUGDUP_ALERT_CNT","DRUGDUP_ACCEPT_CNT","DRUGALLERGY_ALERT_CNT","DRUGALLERGY_ACCEPT_CNT","OTHER_DISCERN_ALERT_CNT","OTHER_DISCERN_ACCEPT_CNT","REV_ADMIN_IND","PTT_COLL_RESULT_STATUS","PTT_COLL_RESULT_VAL","PTT_COLL_RESULT_UNITS","PTT_COLL_UNIQUE_TO_ADMIN","PTT_COLL_ORDER_DT_TM","PTT_COLL_OD_REQ_START_DT_TM","PTT_COLL_OD_PRIORITY","PTT_COLL_OD_SPECIMEN_TYPE","PTT_COLL_OD_NURSECOLLECT","PTT_COLL_OA_COLLECTED","PTT_COLLECTED_BY_UN","PTT_COLLECTED_BY_POSITION","PTT_COMP_RESULT_STATUS","PTT_COMP_RESULT_VAL","PTT_COMP_RESULT_UNITS","DX_LIST","DX_PRIORITY_LIST","DX_TYPE_LIST"]]
myDemographicData.ENCNTR_ID= myDemographicData.ENCNTR_ID.str.split(".",expand=True)[0]
myDemographicData.ADMIN_EVENT_ID= myDemographicData.ADMIN_EVENT_ID.str.split(".",expand=True)[0]
#myComplianceMHGR = pd.read_csv("C:/Users/an052283/Desktop/MHGR_DC_Compliance_by_Admin.csv",dtype={'Admin Event Id':str}).drop_duplicates()
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
#myMilestoneData.dropna(subset=["ADJUSTMENT_TIME_COMPLIANCE","ADJUSTMENT_TIME_COMPLIANCE","DOSING_COMPLIANCE"], how='all',inplace=True)
#myMilestoneData.dropna(subset=["adjustmentCompliance","collectionCompliance","dosingCompliance"], how='all',inplace=True)
#myMilestoneData.dropna(subset=["adjustmentCompliance","collectionCompliance","dosingCompliance"], how='all',inplace=True)
#myMilestoneData.dropna(subset=["dosingCompliance"], how='all',inplace=True)
myMilestoneData.dropna(subset=["collectionCompliance","adjustmentCompliance","dosingCompliance"], how='any',inplace=True)
myMilestoneData.sort_values(by="TIMESTAMP",inplace=True)

traceCompliances=myMilestoneData.groupby(['CLIENT','ENCNTR_ID','PATHWAY_CATALOG_ID']).apply(build_trace_compliance).reset_index()
#traceCompliances.to_csv('C:/Users/an052283/Desktop/MHGR_mo_hep_trace_compliances.tsv',sep='\t',doublequote=True,index=True)

#LOGIC

#milestone names have underscore "_" in names - should probably use camel notation in future but for time's sake use this workaround for now
punctuationNoUnderscore= punctuation.replace("_","")

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

X_train = sequences[:round(traceCompliances.shape[0]/2), :]
#y_train = traceCompliances.collectionCompliancePercent[:round(traceCompliances.shape[0]/2)].values
#y_train = traceCompliances.adjustmentCompliancePercent[:round(traceCompliances.shape[0]/2)].values
y_train = traceCompliances.dosingCompliancePercent[:round(traceCompliances.shape[0]/2)].values
X_test = sequences[round(traceCompliances.shape[0]/2):, :]
#y_test = traceCompliances.collectionCompliancePercent[round(traceCompliances.shape[0]/2):].values
#y_test = traceCompliances.adjustmentCompliancePercent[round(traceCompliances.shape[0]/2):].values
y_test = traceCompliances.dosingCompliancePercent[round(traceCompliances.shape[0]/2):].values

np.random.seed(123) # for reproducibility

# Step 1: Defining multilayer RNN cells
# Step 2: Defining the initial states for the RNN cells
# Step 3: Creating the recurrent neural network using the RNN cells and their states

## Train:

n_words = max(list(word_to_int.values())) + 1

headline_data = np.random.randint(1, sequence_length + 1, size=(32, sequence_length))
additional_data = np.random.randint(1, sequence_length + 1, size=(32, sequence_length))
labels = np.random.randint(0, 1 + 1, size=(32, 1))

#OLD
# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
#NEW
# Headline input: meant to receive sequences of "sequence_length" integers, between 1 and n_words.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(sequence_length,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
#x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
x = Embedding(output_dim=512, input_dim=10000, input_length=sequence_length)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

#Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly
#Even though the main loss will be much higher in the model.
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

#At this point, we feed into the model our auxiliary input data by concatenating it with the LSTM output:
auxiliary_input = Input(shape=(sequence_length,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
#This defines a model with two inputs and two outputs:
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

#We compile the model and assign a weight of 0.2 to the auxiliary loss.
#To specify different loss_weights or loss for each different output, you can use a list or a dictionary.
#Here we pass a single loss as the loss argument, so the same loss will be used on all outputs.

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

#We can train the model by passing it lists of input arrays and target arrays:
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
#Since our inputs and outputs are named (we passed them a "name" argument), we could also have compiled the model via:
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)

#headline_data = np.random.randint(1, 10000 + 1, size=(32, 100))
#additional_data = np.random.randint(1, 10000 + 1, size=(32, 100))
#labels = np.random.randint(0, 1 + 1, size=(32, 1))

