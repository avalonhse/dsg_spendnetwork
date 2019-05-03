#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# Import the Divergence Results Table - Result from test data!
df_similarity = pd.read_csv('TopicModel_test/distance0.csv', index_col=0)
# Transpose the matrix to have the company name as header and contract_id as 
#df_similarity = df_similarity.transpose()


# In[6]:


df_similarity.shape


# In[7]:


df_similarity.head(2)


# In[8]:


# Read only the contract of the test dataset!
contract_testdata = pd.read_csv('0.csv')


# In[9]:


contract_testdata.head(2)


# In[10]:


# Extract the ocid (contract_id) and supplier names (company_name) from the contract test table!
contract = contract_testdata['contract_id']
suppliers = contract_testdata['company_name']
suppliers = [x.lower() for x in suppliers] 

# Create a new dataframe with only the contract_id and supplier name!
new_df = pd.DataFrame()
new_df['contract_id'] = contract
new_df['supplier_name'] = suppliers
new_df.head(2)


# In[11]:


# Create an empty dataframe to store the values 1 if there is a match between company and contract and value 0, if there is not match between company and contract! 
df_match = pd.DataFrame(index = df_similarity.index, columns = list(df_similarity))


# In[12]:


# Fill the cells of the empty dataframe with 1 for the company that have win the contract
for each in new_df.index:
    contract_id = new_df.iloc[each]['contract_id']
    supplier_name = new_df.iloc[each]['supplier_name']
    df_match.loc[supplier_name,contract_id,] = 1
    
# fill the cells of the empty matrix with 0 for the company that does not have win the contract
df_match = df_match.fillna(0)


# In[13]:


df_match.head(2)


# In[14]:


# Save the company/contract match to a csv file. 
df_match.to_csv('testdata_supplier_contract_match.csv')


# In[16]:


# Calculate the total number of contracts for each of the suppliers!
total_contract_per_suppliers = list(df_match.sum(axis=1))
print(total_contract_per_suppliers)


# In[17]:


# Check if the shape of both dataframe are the same!
print(df_match.shape)
print(df_similarity.shape)


# In[18]:


# Transform the dataframe (df_match and df_similarity) to numpy array to calculate the number of retrieval!
supplier_match = df_match.values
similarity_numpy = df_similarity.values


# In[165]:


"""Create a function that retrieves the true number of contracts a supplier have
been awarded. """

def score_for_supplier_at_threshold(similarity_table, truth_contract, threshold):
    scores = []
    for supplier_idx in range(len(similarity_table)):
        # apply thresholding
        filtered_supplier = similarity_table[supplier_idx] < threshold

        #compare with corresponding company in truth table
        filtered_supplier == truth_contract[supplier_idx]
        matches = np.logical_and(filtered_supplier,truth_contract[supplier_idx])
        scores.append(np.sum(matches))
    return scores


# In[174]:


"""Create a function that retrieves the false number of contracts a supplier have
been awarded. """

def false_score_for_supplier(similarity_table, truth_contract, threshold):
    # calculate the number of contract that are below the threshold. 
    false_scores = []
    for supplier_idx in range(len(similarity_table)):
        # apply thresholding
        filtered_supplier = similarity_table[supplier_idx] >= threshold

        #compare with corresponding company in truth table
        filtered_supplier == truth_contract[supplier_idx]
        matches = np.logical_and(filtered_supplier,truth_contract[supplier_idx])
        false_scores.append(np.sum(matches))
    return false_scores


# In[175]:


# Have a look if the function is working!
test_false_threshold2 = false_score_for_supplier(similarity_numpy, supplier_match,2)
print(test_false_threshold2)


# In[166]:


# Have a look if the function is working!
test_threshold2 = score_for_supplier_at_threshold(similarity_numpy, supplier_match,2)
print(test_threshold2)


# In[206]:


"""Create a function that return a table with the true retrieval score for all companies. 
True Values of contract, suppliers with contract above threshold (threshold from 0 to 12)!"""

def evaluation_threshold_true():
    similarity_table = similarity_numpy
    truth_contract = supplier_match
    thresholds = np.arange(0,12,0.1)
    table = []
    for thresh in thresholds:
        table.append(score_for_supplier_at_threshold(similarity_table, truth_contract, thresh))
    return table


# In[207]:


"""Create a function that create a csv file with the accuracy score for all companies 
for the range of threshold from 0 to 10."""
import csv

def evaluation_threshold_false():
    similarity_table = similarity_numpy
    truth_contract = supplier_match
    thresholds = np.arange(0,12,0.1)
    table = []
    for thresh in thresholds:
        table.append(false_score_for_supplier(similarity_table, truth_contract, thresh))
    return table


# In[208]:


test = np.arange(0,12,0.1)
test = np.around(test, decimals=1)
test
len(test)


# In[209]:


# Store the true retrieval values in a dataframe. 
evaluation_true = pd.DataFrame(evaluation_threshold_true())
evaluation_true['threshold'] = test
evaluation_true = evaluation_true.set_index(['threshold'])
evaluation_true.head()


# In[210]:


evaluation_true = evaluation_true.transpose()
#evaluation.columns = test
evaluation_true.head()


# In[211]:


# Store the false retrieval values in a dataframe. 
evaluation_false = pd.DataFrame(evaluation_threshold_false())
evaluation_false['threshold'] = test
evaluation_false = evaluation_false.set_index(['threshold'])
evaluation_false = evaluation_false.transpose()
evaluation_false.head()


# In[180]:


# Add the name of the suppliers to the evaluation dataframe. 
# Retrieve the suppliers name from the df_similairty dataframe index!
suppliers_name = df_similarity.index
suppliers_name_list = list(suppliers_name)


# In[212]:


# Rename the columns of the evaluation dataframe with the supplier names.
evaluation_true = evaluation_true.transpose()
evaluation_true.columns = suppliers_name_list

evaluation_false = evaluation_false.transpose()
evaluation_false.columns = suppliers_name_list


# In[242]:


evaluation = 1/(1+(np.exp(evaluation_false - evaluation_true)))


# In[243]:


evaluation.head(2)


# In[244]:


evaluation['retrieval_score_mean'] = evaluation.mean(axis=1)
evaluation['threshold'] = evaluation.index
evaluation.head(2)


# In[245]:


evaluation.to_csv('evaluation.csv')


# In[246]:


# Draw the plot for all suppliers

threshold = evaluation['threshold']
all_suppliers_retrieval_average = evaluation['retrieval_score_mean']

fig = plt.figure(figsize=(14,7))

sns.set(style='darkgrid', rc={"lines.linewidth": 0.8})
plt.plot(threshold,all_suppliers_retrieval_average,marker=".")
plt.xlabel('Threshold of distance measure',fontsize = 16)
plt.ylabel('Retrieval Score',fontsize = 16)
plt.title('Evaluation for all suppliers',fontsize = 18)

fig.savefig('Evaluation_plot_threshold_by1.png',dpi=fig.dpi)


# In[ ]:


# Plot for single company

threshold = evaluation['threshold']
all_suppliers_retrieval_average = evaluation.iloc[:,0]

fig = plt.figure(figsize=(8,5))

sns.set(style='darkgrid', rc={"lines.linewidth": 0.8})
plt.plot(threshold,all_suppliers_retrieval_average,marker=".")
plt.xlabel('Distance - Threshold')
plt.ylabel('Retrieval Value Score')
plt.title('Evaluation for all Suppliers')

fig.savefig('Evaluation_plot_threshold_by1.png',dpi=fig.dpi)

