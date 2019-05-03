from __future__ import division
import pickle
import io
import os, sys
import csv
import re
import nltk
from string import punctuation

def removePunctuation(text):
    '''
    Removes punctuation, changes to lower case and strips leading and trailing
    spaces.

    Args:
        text (str): Input string.

    Returns:
        (str): The cleaned up string.
    '''
    text.strip()
    return ''.join(c for c in text.encode('ascii', 'ignore') if c not in punctuation or c in ['#','@','.','\n'])

def serialize(obj, ser_filename, isSerializingList=True):
    '''
    Saves a python object to disk.

    If the object being dealt with is a list, the contents of thenew list get
    added to the existing serialized list. Otherwise, the new object ovewrites
    the old one.

    Args:
        obj: Object to save.
        ser_filename: Filename to save object with on disk.
        isSerializingList: Boolean denoting whether object to be saved is a list
                            or not.
    '''
    if(isSerializingList):
        # If pre-existing serialization, get its list representation and add new one to it
        if(os.path.isfile(ser_filename)):
            stored_list = unserialize(ser_filename)
            stored_list.extend(obj)
            f = open(ser_filename, 'wb')
            pickle.dump(stored_list, f)
            f.close()
        else:
            f = open(ser_filename, 'wb')
            pickle.dump(obj, f)
            f.close()
    else:
        f = open(ser_filename, 'wb')
        pickle.dump(obj, f)
        f.close()

def unserialize(ser_filename):
    '''
    Loads a python object from disk.

    Returns:
        The python object at the specified path or None if none is found.
    '''
    if(not os.path.isfile(ser_filename)):
        return None
    else:
        f = open(ser_filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

def map_node_to_dataset_id(content_embedding_data):
    '''
    This functions returns a dictionary which maps a graph node id (which is
    just its position in a list) to the contract or supplier id for that node.
    i.e. The dictionary is of the form {0:'<ID_1>', 1:'<ID_1>', ... , <NODE_N>:'<ID_N>'}

    @param adjmat adjacency matrix as produced by the "adjacency_matrix(similarity_matrix, threshold)"
    function and seen in the corresponding output csv.
    @param content_embedding_data content embeddings as produced by either the "get_train_embeddings()"
    or "get_test_embeddings()" and the corresponding csv files.
    @return mapping dict
    '''
    mapping = {}
    for idx in range(len(content_embedding_data)):
        mapping[idx] = content_embedding_data[idx]['id']

    return mapping
