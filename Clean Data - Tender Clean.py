#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


tender_text = pd.read_csv('ocds_tenders_clean_new.csv')


# In[5]:


tender_text.shape


# In[6]:


tender_text.columns


# In[7]:


tender_text.head(2)


# In[10]:


# Cleaning Text Data
import string
import nltk
import re

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[11]:


# import nltk and download punctuations
nltk.download('punkt')


# In[12]:


# Take the relevant text of the suppliers data
tender1 = tender_text['title']
tender2 = tender_text['description']


# In[16]:


# Join text together to process to cleaning
tender_text_join = pd.DataFrame(tender1 + tender2)

# Rename tender_text
tender_text_join = tender_text_join.rename(columns={0:"tender_text"})


# In[17]:


tender_text_join.head(2)


# In[18]:


# Remove all the '\n' line-brakes
tender_text_join.tender_text = tender_text_join.tender_text.str.replace(r'\n',' ')


# In[21]:


# Remove all the https
tender_text_join.tender_text = tender_text_join.tender_text.str.replace(r'http\S+','')


# In[23]:


# Remove 'non-ascii' characters
tender_text_join.tender_text = tender_text_join.tender_text.str.replace(r'[^\x00-\x7F]',' ')
# Make all the text lowercase
tender_text_join.tender_text = tender_text_join.tender_text.str.lower()


# In[25]:


# Remove numbers 
tender_text_join.tender_text = tender_text_join.tender_text.str.replace(r'\d+','')


# In[27]:


# Remove punctuations
tender_text_join['clean_text'] = tender_text_join.tender_text.apply(
    lambda x:str(x).translate(str.maketrans('','', string.punctuation)))


# In[31]:


# Tokenise all of the clean text
tender_text_join['token_text'] = tender_text_join.clean_text.apply(word_tokenize)


# In[32]:


tender_text_join.head(4)


# In[34]:


tender_text_join['token_text'][1]


# In[35]:


# Save the clean text for supplier in a csv file!
tender_text_join.to_csv('tender_text_clean.csv')

