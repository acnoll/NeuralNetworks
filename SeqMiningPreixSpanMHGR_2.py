import pandas as pd
from string import punctuation
import numpy as np
import math
from collections import Counter
from prefixspan import PrefixSpan

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
    orderWeight = round(pd.to_numeric(myData['ORDER_WEIGHT'], errors='coerce').mean())
    gender=myData['SEX'].iloc[0]
    adminMedService=myData['ADMIN_MED_SERVICE'].iloc[0]
    adminFacility = myData['ADMIN_FACILITY'].iloc[0]
    adminUnit = myData['ADMIN_UNIT'].iloc[0]
    maritalStatus= myData['MARITAL_STATUS'].iloc[0]
    language = myData['LANGUAGE'].iloc[0]
    race = myData['RACE'].iloc[0]
    adjustmentCompliancePercent = round(pd.to_numeric(myData.adjustmentCompliance, errors='coerce').mean())
    collectionCompliancePercent = round(pd.to_numeric(myData.collectionCompliance, errors='coerce').mean())
    dosingCompliancePercent = round(pd.to_numeric(myData.dosingCompliance, errors='coerce').mean())
    c = ['race','maritalStatus','language','adminAgeYrs','adminMedService','adminFacility','adminUnit','gender','adjustmentCompliancePercent', 'collectionCompliancePercent', 'dosingCompliancePercent','orderWeight']
    return pd.Series([race,maritalStatus,language,adminAgeYrs,adminMedService,adminFacility,adminUnit,gender,adjustmentCompliancePercent,collectionCompliancePercent, dosingCompliancePercent,orderWeight],index=c)

#DATA MUNGING
#create some values for conversion
nonCompliantAdjustment=["Noncompliant","Noncompliant 30min HOLD","Noncompliant 60min HOLD","Noncompliant 120min HOLD"]
compliantAdjustment=["Compliant Adjustment Time","Compliant 30min HOLD","Compliant 60min HOLD","Compliant 120min HOLD"]
compliantCollection=["Compliant Collection Time"]
nonCompliantCollection=["Early/Late Collection Time","No Lab Collected w/in 12 hrs"]
nonCompliantDosing=["Noncompliant"]
compliantDosing=["Compliant Dose","Compliant Supratherapeutic Turnoff"]

#myMilestoneData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_hep_milestone_110117_110118_009.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},error_bad_lines=False,sep='\t',encoding = "ISO-8859-1").drop_duplicates()
myMilestoneData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_heparin_milestone_110117_011819_014.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},error_bad_lines=False,sep='\t',encoding = "ISO-8859-1").drop_duplicates()
#myMilestoneData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_heparin_milestone_092318_011819_014.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},error_bad_lines=False,sep='\t',encoding = "ISO-8859-1").drop_duplicates()
myMilestoneData[['TIMESTAMP']]=myMilestoneData[['TIMESTAMP']].apply(pd.to_datetime,format='%d-%b-%Y %H:%M:%S',errors='ignore')
myMilestoneData.ENCNTR_ID= myMilestoneData.ENCNTR_ID.str.split(".",expand=True)[0]
myMilestoneData.ADMIN_EVENT_ID= myMilestoneData.ADMIN_EVENT_ID.str.split(".",expand=True)[0]
myMilestoneData=myMilestoneData[-myMilestoneData.MILESTONE_NAME.str.contains("ptt|protamine_admin|heparin_admin_charted|heparin_begin_bag")]
#myDemographicData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_hep_protocol_110117_110118_009.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},sep='\t',encoding = "ISO-8859-1").drop_duplicates()
myDemographicData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_heparin_protocol_110117_011819_014.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},sep='\t',encoding = "ISO-8859-1").drop_duplicates()
myDemographicData.RACE.loc[pd.isnull(myDemographicData.RACE)]="Unknown"
#myDemographicData = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_heparin_protocol_092318_011819_014.tsv",dtype={'ENCNTR_ID':str,'ADMIN_EVENT_ID':str},sep='\t',encoding = "ISO-8859-1")

#subset for what we need
myDemographicData=myDemographicData[["CLIENT","ADMIN_AGE_YRS","LANGUAGE","SEX","MARITAL_STATUS","RACE","RELIGION","ENCNTR_ID","REG_PRSNL_UN","REG_MED_SERVICE","REG_FACILITY","REG_UNIT","DISCH_DISPOSITION","PATHWAY_CATALOG_ID","ORDER_SET","ORDER_MNEMONIC","CATALOG_CD","ORDER_MED_SERVICE","ORDER_FACILITY","ORDER_UNIT","ORDER_SENTENCE","ORDER_WEIGHT","ORDER_WEIGHT_UNITS","ORDERING_PROVIDER_UN","ORDERING_PROVIDER_POSITION","ADMIN_PRSNL_ID","ADMIN_PRSNL_UN","ADMIN_PRSNL_POSITION","ADMIN_MED_SERVICE","ADMIN_FACILITY","ADMIN_UNIT","ADMIN_UNIT_DESCRIPTION","ADMIN_ENCNTR_TYPE","ADMIN_CATALOG_CD","ADMIN_EVENT_CD","ADMIN_EVENT_CD_DISP","ADMIN_EVENT_ID","ADMIN_PARENT_EVENT_ID","ADMIN_INFUSION_RATE","ADMIN_INFUSION_UNITS","ADMIN_RESULT_STATUS","ADMIN_EVENT_TAG","ADMIN_VERIFIED_PRSNL_ID","ADMIN_WITNESS_BY_UN","CHARTING_APP","PT_SCANNED_IND","MED_SCANNED_IND","PUMP_PROGRAMMING_IND","PUMP_PROG_ELIGIBLE_IND","RX_ROUTE","PENDING_VERIFY_AT_ADMIN_IND","VERIFY_STATUS_AT_ADMIN","DRC_ALERT_CNT","DRC_ACCEPT_CNT","DRUGDRUG_ALERT_CNT","DRUGDRUG_ACCEPT_CNT","DRUGDUP_ALERT_CNT","DRUGDUP_ACCEPT_CNT","DRUGALLERGY_ALERT_CNT","DRUGALLERGY_ACCEPT_CNT","OTHER_DISCERN_ALERT_CNT","OTHER_DISCERN_ACCEPT_CNT","REV_ADMIN_IND","PTT_COLL_RESULT_STATUS","PTT_COLL_RESULT_VAL","PTT_COLL_RESULT_UNITS","PTT_COLL_UNIQUE_TO_ADMIN","PTT_COLL_ORDER_DT_TM","PTT_COLL_OD_REQ_START_DT_TM","PTT_COLL_OD_PRIORITY","PTT_COLL_OD_SPECIMEN_TYPE","PTT_COLL_OD_NURSECOLLECT","PTT_COLL_OA_COLLECTED","PTT_COLLECTED_BY_UN","PTT_COLLECTED_BY_POSITION","PTT_COMP_RESULT_STATUS","PTT_COMP_RESULT_VAL","PTT_COMP_RESULT_UNITS","DX_LIST","DX_PRIORITY_LIST","DX_TYPE_LIST"]]
myDemographicData.ENCNTR_ID= myDemographicData.ENCNTR_ID.str.split(".",expand=True)[0]
myDemographicData.ADMIN_EVENT_ID= myDemographicData.ADMIN_EVENT_ID.str.split(".",expand=True)[0]
myComplianceMHGR = pd.read_csv("C:/Users/an052283/Desktop/mhgr_dc_heparin_014_compliance_v29.csv",dtype={'Admin Event Id':str}).drop_duplicates()
#myComplianceMHGR = pd.read_csv("C:/Users/an052283/Desktop/MHGR_DC V29 Compliance by Admin (w Mid).csv",dtype={'Admin Event Id':str}).drop_duplicates()

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
punctuationNoUnderscore= punctuation.replace("_","")

counts = Counter()

for i,trace in enumerate(traceCompliances['trace']):
    text = ''.join([c if c not in punctuationNoUnderscore else ' '+c+' ' \
                    for c in trace]).lower()
    traceCompliances.loc[i,'trace'] = text
    counts.update(text.split())

## Create a mapping: Map each unique word to an integer
word_counts = sorted(counts, key=counts.get, reverse=True)
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
mapped_traces = []
for trace in traceCompliances['trace']:
    mapped_traces.append([word_to_int[word] for word in trace.split()])
## Define fixed-length sequences: Use the last 200 elements of each sequence; if sequence length < 200: left-pad with zeros
sequence_length = traceCompliances.trace.map(lambda x: x.count(' ')).max() +2  ## sequence length (or T in our formulas)
sequences = np.zeros((len(mapped_traces), sequence_length), dtype=int)
for i, row in enumerate(mapped_traces):
    trace_arr = np.array(row)
    sequences[i, -len(row):] = trace_arr[-sequence_length:]  #sequence length versus len(row) ?
#TRACE DATA
trainStop=round(traceCompliances.shape[0]*0.6)
testStop=round(traceCompliances.shape[0]*0.8)
x_sequence_train = sequences[:trainStop]
y_sequence_train = traceCompliances.dosingCompliancePercent[:trainStop].values
x_sequence_test = sequences[trainStop:testStop]
y_sequence_test = traceCompliances.dosingCompliancePercent[trainStop:testStop].values
x_sequence_valid = sequences[testStop:]
y_sequence_valid = traceCompliances.dosingCompliancePercent[testStop:].values
#STATIC DATA
static_trace_data =myMilestoneData.groupby(['CLIENT','ENCNTR_ID','PATHWAY_CATALOG_ID']).apply(build_static_compliance).reset_index()
static_trace_data=pd.get_dummies(static_trace_data[['adminMedService','adminFacility','adminUnit','race','maritalStatus','language','gender']])
x_static_train = static_trace_data[:trainStop]
x_static_test = static_trace_data[trainStop:testStop]
x_static_valid = static_trace_data[testStop:]
#general config/setup
np.random.seed(123)# for reproducibility
n_words = max(list(word_to_int.values())) + 1

# db = [
#     [0, 1, 2, 3, 4],
#     [1, 1, 1, 3, 4],
#     [2, 1, 2, 2, 0],
#     [1, 1, 1, 2, 2],
# ]
#
# db =[
# ['a','b','c','d','e'],
# ['b','b','b','d','e'],
# ['c','b','c','c','a'],
# ['b','b','b','c','c'],
# ]
#
# db =[
# ['hello','hola','hallo','gday','ningho'],
# ['hola','hola','hola','gday','ningho'],
# ['hallo','hola','hallo','hallo','hello'],
# ['hola','hola','hola','hallo','hallo'],
# ]
# ps = PrefixSpan(db)
# print(ps.frequent(2))
# print(ps.topk(5,closed=True))
# print(ps.frequent(2, closed=True))
# print(ps.topk(5, key=lambda patt, matches: sum(len(db[i]) for i, _ in matches)))
# print(ps.topk(5, filter=lambda patt, matches: matches[0][0] > 0))
# db[3]

db=traceCompliances.trace.str.split(' , ').tolist()
ps = PrefixSpan(db)
#topk_10=ps.topk(10,generator=True)
#only include sequences that have representation in at least 75% of the traces?
freq=ps.frequent(round(len(db)*0.1),generator=True)
#get median length and take 90% of it to allow for smaller subsequences and make that a cutoff?
medianLength=round((pd.DataFrame(len(i) for i in db).median()*0.5).loc[0])
minStepsHighFreqSeqs = [seq for seq in freq if len(seq[1])>=medianLength]
traces = [",".join(trace) for trace in db]
freqSeqs = [",".join(seq[1]) for seq in freq]

#TOMORROW NEED TO
#1) ITERATE THROUGH THE TRACES AND SEARCH FOR EACH OF THE minStepsHighFreqSeqs and populate A DATAFRAME WITH ROWS BEING THE TRACE AND COLUMNS BEING 0/1 IF HAD THE SUBSEQUENCE OR NOT

i=0
j=0
subseqTraceMtx = np.zeros((len(traces),len(freqSeqs)))
for trace in traces:
 for seq in freqSeqs:
  if seq.find(trace)>=0:
   subseqTraceMtx[i,j] = 1
  j+=1
 i+=1
 j=0

a=pd.DataFrame(subseqTraceMtx)
a.sum(axis=1)
import seaborn as sns
ax = sns.boxplot(x=a.sum(axis=1))
import scipy.stats as stats
import pylab
stats.probplot(a.sum(axis=1), dist="norm", plot=pylab)
pylab.show()

#2) DETERMINE BEST WAY TO CLASSIFY, MAYBE USE SIMPLE LR APPROACH

#3) INTRODUCE THE STATIC DEMOGRAPHIC DATA
#4) PUT THROUGH lr again AND SEE WHAT COEFFICIENTS ARE FOR EACH VARIABLE
#4.5) see what prediction power is for model
#5) CONSIDER OTHER CLASSIFICATION TECHNIQUES SUCH AS DECISION TREE


# coverage = [[] for i in range(len(db))]
# def cover(patt, matches):
#     for i, _ in matches:
#         coverage[i] = max(coverage[i], patt, key=len)
#testStop=ps.frequent(round(len(db)*0.5), callback=cover,generator=True)