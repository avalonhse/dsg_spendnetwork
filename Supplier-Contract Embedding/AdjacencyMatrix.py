import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import csv
import EntityEmbedding
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk.tokenize
import Helper

def cos_similarity(a, b):
    return np.inner(a, b)/(norm(a)*norm(b))


def similarity_matrix(suppliers, contracts):
    '''
    Takes two lists: one for suppliers and another for contracts,
    and returns a (cosine) similarity matrix for suppliers and contracts.

    INPUTS:
        - suppliers: numpy array of supplier vectors
        - contracts: numpy array of contract vectors

    OUTPUTS:
        - sigma: similarity matrix of suppliers and contracts.
    '''
    sigma = np.empty([len(suppliers), len(contracts)])
    for i in range(len(suppliers)):
        for j in range(len(contracts)):
            sigma[i, j] = cos_similarity(suppliers[i,:], contracts[j,:])
    return sigma


def adjacency_matrix(similarity_matrix, threshold, outfilename):
    '''
    Takes a similarity matrix, and returns an adjacency matrix

    INPUTS:
        - similarity_matrix: a matrix with cosine similarities
        - threshold: threshold above which there is a link between supplier and contract

    OUTPUTS:
        - adjacency_matrix (filetype:csv): first column are the supplier nodes,
                                           second column are contract nodes,
                                           third columns are similarity measures between
    '''
    adjacency_matrix = dict()
    num_suppliers = similarity_matrix.shape[0]
    for row in range(similarity_matrix.shape[0]):
        adjacency_matrix[row] = (np.argwhere(similarity_matrix[row,:]>threshold) +\
                                num_suppliers).reshape(1,-1)[0].tolist()

    number_edges = sum(map(len, adjacency_matrix.values()))
    number_rows = similarity_matrix.shape[0]+similarity_matrix.shape[1]

    with open(outfilename, 'w') as f:
        f.write("{},{}\n".format(number_rows, number_edges))
        for key in adjacency_matrix.keys():
            for i in range(len(adjacency_matrix[key])):
                similarity = similarity_matrix[key][i - num_suppliers]
                f.write("{},{},{}\n".format(key,adjacency_matrix[key][i], similarity))

def filter_adj_mat(infilename, threshold, outfilename):
    newCSVRows = []
    oldCSVRows = []
    with open(infilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        newCSVRows.append(reader.next())
        for row in reader:
            if float(row[2]) > threshold:
                newCSVRows.append(row)
            oldCSVRows.append(row)
    # check to see which rows where missed because there was no similarity satisfying the thresholding
    # flatten old csv data to a dict where each key is the supp id and the value is a list of corresponding contracts
    oldDict = {}
    for old_row in oldCSVRows:
        #print old_row
        if(old_row[0] not in oldDict.keys()):
            oldDict[old_row[0]] = []
        oldDict[old_row[0]].append([float(old_row[1]),float(old_row[2])])
    for key in oldDict.keys():
        rowDoesExist = False
        for new_row in newCSVRows:
            if key == new_row[0]:
                rowDoesExist = True
                break
        if not rowDoesExist:
            newCSVRows.append(old_row)
    print newCSVRows[4]
    with open(outfilename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for row in newCSVRows:
            writer.writerow(row)

def generate_matrices():
    if __name__ == "__main__":
        partitions = EntityEmbedding.get_train_embeddings(False,False)

        p = 0
        for doc2vec, content_embeddings_data, entities in partitions:
            print p
            supplier_vecs_list = []
            tender_vecs_list = []
            i = 0
            for emb_data in content_embeddings_data:
                if(emb_data['type'] == 'SUPPLIER'):
                    supplier_vecs_list.append(emb_data['vec'])
                elif(emb_data['type'] == 'TENDER'):
                    tender_vecs_list.append(emb_data['vec'])

            # Create numpy arrays
            S = np.array(supplier_vecs_list)
            T = np.array(tender_vecs_list)
            print S
            print S.shape
            print T
            print T.shape
            print('\n\n')

            # Obtain similarity matrix
            sim = similarity_matrix(S, T)
            print sim
            print('\n\n')

            #Construct adjacency matrix
            threshold = 0
            G = adjacency_matrix(sim, threshold, 'out_train/adjacency_matrix_'+str(p)+'.csv')

            filter_adj_mat('out_train/adjacency_matrix_'+str(p)+'.csv', 0.9, 'out_train/filtered_adjacency_matrix_'+str(p)+'.csv')

            # Create mapping from node id to dataset id (i.e. supplier and contract id)
            mapping = Helper.map_node_to_dataset_id(content_embeddings_data)
            with open('out_train/node_id_mapping_'+str(p)+'.csv', 'wb') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['node_id', 'dataset_id'])
                for node_id in mapping.keys():
                    writer.writerow([node_id, mapping[node_id]])

            p += 1
