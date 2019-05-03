from __future__ import division
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk.tokenize
import csv, os, re
import scipy.spatial.distance as distance
import collections
import Helper
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# with open('data/entity_website_text.csv') as csvfile:
EPOCHS = 100
EMBEDDING_DIM = 200
ALPHA = 0.025
DISPLAY_INTERVAL = 20

class Entity(object):
    '''
    Abstraction for an entity in our system. Could be either a "supplier" or a
    "tender" (which is in essence a contract as its from the contract table).
    '''
    def __init__(self, id=None, name=None, text=None, type=None):
        '''
        Constructor for Entity.

        @param id: Either contract_id in case where Entity is a contract, or
        company_name in case where Entity is a supplier.
        @param name: Either company_name in the case where Entity is a Supplier, or empty.
        @param text: Either contract_text or company_text.
        @param type: value of either 'SUPPLIER' or 'TENDER', depending on whether the Entity is
        a supplier or tender (synonymous with contract for our intents and purposes).
        '''
        self.id = id
        self.name = name
        self.text = text
        self.type = type

def remove_duplicate_ids(list_of_dicts):
    '''
    Removed duplicate ids from a list of dictionaries and returns a new dictionary.
    '''
    list_of_dicts_duplicates = {}
    for i in xrange(len(list_of_dicts)):
        list_of_dicts_duplicates[list_of_dicts[i]['company_name']] = list_of_dicts[i]
    return list_of_dicts_duplicates.values()

def remove_duplicate_suppliers(list_of_suppliers):
    list_of_suppliers_duplicates = {}
    for i in xrange(len(list_of_suppliers)):
        list_of_suppliers_duplicates[list_of_suppliers[i].name] = list_of_suppliers[i]
    return list_of_suppliers_duplicates.values()

def remove_duplicate_from_csv(filename):
    '''
    Removes duplicate rows from a csv based on the 'company_name' field and writes
    a new csv to replace the input.
    '''
    entries = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entries.append(row)
    entries = remove_duplicate_ids(entries)
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, entries[0].keys())
        writer.writeheader()
        for row in entries:
            writer.writerow(row)

def load_supplier_and_tender_data():
    '''
    Loads supplier and tender information using the training set.

    @Returns
    Two lists of Entity objects, where one list contains supplier entities and the
     other contains tender (contract) entities.
    '''
    partitions = []


    for filename in sorted(os.listdir('extracted_data/training_sets')):
        supplier_objects = []
        tender_objects = []

        if(filename.endswith('.csv')):
            #remove_duplicate_from_csv('extracted_data/training_sets'+"/"+filename)
            with open('extracted_data/training_sets'+"/"+filename) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    newSupplier = Entity(row['company_name'].decode('utf-8'), row['company_name'].decode('utf-8'), preprocess_text(row['company_text'].decode('utf-8')+row['company_info'].decode('utf-8')), 'SUPPLIER')
                    newTender = Entity(row['contract_id'].decode('utf-8'), '', preprocess_text(row['contract_text'].decode('utf-8')), 'TENDER')
                    supplier_objects.append(newSupplier)
                    tender_objects.append(newTender)
                partitions.append((remove_duplicate_suppliers(supplier_objects),tender_objects))

    return partitions

def load_test_data():
    '''
    Loads supplier and tender information using both the test set.

    @Returns
    Two lists of Entity objects, where one list contains supplier entities and the
     other contains tender (contract) entities.
    '''
    partitions = []

    for filename in sorted(os.listdir('extracted_data/testing_sets')):
        supplier_objects = []
        tender_objects = []

        if(filename.endswith('.csv')):
            print filename
            #remove_duplicate_from_csv('extracted_data/testing_sets'+"/"+filename)
            with open('extracted_data/testing_sets'+"/"+filename) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    newSupplier = Entity(row['company_name'].decode('utf-8'), row['company_name'].decode('utf-8'), preprocess_text(row['company_text'].decode('utf-8')+row['company_info'].decode('utf-8')), 'SUPPLIER')
                    newTender = Entity(row['contract_id'].decode('utf-8'), '', preprocess_text(row['contract_text'].decode('utf-8')), 'TENDER')
                    supplier_objects.append(newSupplier)
                    tender_objects.append(newTender)
            partitions.append((remove_duplicate_suppliers(supplier_objects),tender_objects))

    return partitions

def preprocess_text(txt):
    '''
    Helper function to preprocess text
    '''
    # Lower case
    txt = txt.lower()

    # Remove postcodes
    txt = re.sub('[a-z]{1,2}[0-9r][0-9a-z]? [0-9][a-z]{2}', '<POSTCODE>', txt)

    # remove urls
    txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', txt)

    txt = Helper.removePunctuation(txt)

    return txt


def get_train_embeddings(shouldTrain=True, shouldOutput=True):
    '''
    Trains Doc2Vec embeddings for suppliers and tenders (assumed from contact text) and
    outputs them to disk as separate csv files.

    @param shouldTrain Boolean flag controlling whether to train Doc2Vec embeddings
    from scratch or attempt to load a saved model from disk.
    @param shouldOutput Boolean flag controlling whether to output embeddings to
    csv files

    @return (model, supplier_texts, tender_texts) a tuple containing the trained
     gensim Doc2Vec model and the supplier and tender (from contracts) texts.
    '''
    # Extract texts from dataset
    partitions = load_supplier_and_tender_data()
    result_partitions = []

    i = 0
    for supplier_objects, tender_objects in partitions:
        all_entities = list(supplier_objects)
        all_entities.extend(tender_objects)
        print len(all_entities)

        # Tokenize texts
        tagged_data = [TaggedDocument(words=nltk.tokenize.word_tokenize(ent.text.lower()), tags=[str(k)]) for k, ent in enumerate(all_entities)]

        # Build model
        if(shouldTrain):
            model = Doc2Vec(size=100,
                            alpha=ALPHA,
                            min_alpha=0.0025,
                            min_count=1,
                            dm=1,
                            dbow_words=0,
                            iter=300)
            model.build_vocab(tagged_data)

            # Train
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
            model.save("d2v"+str(i)+".model")
            print("Model Saved")
        else:
            model = Doc2Vec.load("d2v"+str(i)+".model")


        # Prepare embedding outputs
        content_embeddings_data = []
        for ent in all_entities:
            vector4text = model.infer_vector(ent.text)
            content_embeddings_data.append({
                'id':ent.id,
                'name':ent.name,
                'vec':vector4text,
                'type':ent.type
            })

        # Output to csv using pandas, so will need to format as dataframe
        if(shouldOutput):
            data = {
                        'id': [ced['id'] for ced in content_embeddings_data],
                        'name': [ced['name'] for ced in content_embeddings_data],
                        'vec': [ced['vec'] for ced in content_embeddings_data],
                        'type': [ced['type'] for ced in content_embeddings_data]
                    }
            df = pd.DataFrame(data, columns= ['id', 'name', 'vec', 'type'])
            df.to_csv('out_train/train_content_embedding_data_0' + str(i) + '.csv', index=None, header=True)
        i += 1
        result_partitions.append((model, content_embeddings_data, all_entities))
    return result_partitions

def get_test_embeddings(shouldOutput=True):
    '''
    Infers Doc2Vec embeddings for suppliers and tenders (assumed from contact text)
    from the test data and outputs them to disk as separate csv files.

    @param shouldTrain Boolean flag controlling whether to train Doc2Vec embeddings
    from scratch or attempt to load a saved model from disk.
    @param shouldOutput Boolean flag controlling whether to output embeddings to
    csv files

    @return (model, supplier_texts, tender_texts) a tuple containing the trained
     gensim Doc2Vec model and the supplier and tender (from contracts) texts.
    '''
    # Extract texts from dataset
    partitions = load_test_data()
    print len(partitions)
    i= 0
    for supplier_objects, tender_objects in partitions:
        all_entities = list(supplier_objects)
        all_entities.extend(tender_objects)
        print(len(all_entities))

        model = Doc2Vec.load("d2v"+str(i)+".model")

        # Prepare embedding outputs
        content_embeddings_data = []
        for ent in all_entities:
            vector4text = model.infer_vector(ent.text)
            content_embeddings_data.append({
                'id':ent.id,
                'name':ent.name,
                'vec':vector4text,
                'type':ent.type
            })

        # Output to csv using pandas, so will need to format as dataframe
        if(shouldOutput):
            data = {
                        'id': [ced['id'] for ced in content_embeddings_data],
                        'name': [ced['name'] for ced in content_embeddings_data],
                        'vec': [ced['vec'] for ced in content_embeddings_data],
                        'type': [ced['type'] for ced in content_embeddings_data]
                    }
            df = pd.DataFrame(data, columns= ['id', 'name', 'vec', 'type'])
            df.to_csv('out_test/test_content_embedding_data_0' + str(i) + '.csv', index=None, header=True)
        i += 1
    #return model, content_embeddings_data, all_entities

def plot_all_embeddings():
    '''
        Plot embeddings for suppliers and contracts in test partition 0
    '''
    #embeds = np.array([[1,2,3], [4,5,6], [1,2,2]])
    _,content_embeddings_data,_ = get_train_embeddings(False,False)[0]
    embeds = []
    for ced in content_embeddings_data:
        embeds.append(ced['vec'])
    n_classes = 5

    # TSNE
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y_TSNE = tsne.fit_transform(embeds)

    # PCA
    Y_PCA = PCA(n_components=2).fit_transform(embeds)

    # Cluster
    kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=n_classes)
    kmeans.fit(Y_TSNE)
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(Y_TSNE)

    fig = plt.figure()
    plt.scatter(Y_TSNE[:, 0], Y_TSNE[:, 1], c=Z, cmap=matplotlib.colors.ListedColormap(['red','blue','yellow','pink','orange']))
    plt.show()

def plot_supplier_tender_embeddings():
    '''
        Plot embeddings for suppliers in test partition 0
    '''
    supplier_objects, tender_objects = load_test_data()[0]
    s_embeds = []
    model = Doc2Vec.load("models/d2v"+str(0)+"_dm100_min25.model")
    for sup in supplier_objects:
        s_embeds.append(model.infer_vector(sup.text))
    n_classes = 5

    # TSNE
    tsne_s = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    YS_TSNE = tsne_s.fit_transform(s_embeds)

    # PCA
    YS_PCA = PCA(n_components=2).fit_transform(s_embeds)

    # Cluster
    kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=n_classes)
    kmeans.fit(YS_TSNE)
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(YS_TSNE)
    fig = plt.figure()
    plt.scatter(YS_TSNE[:, 0], YS_TSNE[:, 1], c=Z, cmap=matplotlib.colors.ListedColormap(['red','blue','orange','yellow','black']))
    plt.show()

def get_train_embeddings_experimental(shouldTrain=True, shouldOutput=True):
    '''
    Trains Doc2Vec embeddings for suppliers and tenders (assumed from contact text) and
    outputs them to disk as separate csv files. Used for experimental training
    attempts and hyperparameter tuning

    @param shouldTrain Boolean flag controlling whether to train Doc2Vec embeddings
    from scratch or attempt to load a saved model from disk.
    @param shouldOutput Boolean flag controlling whether to output embeddings to
    csv files

    @return (model, supplier_texts, tender_texts) a tuple containing the trained
     gensim Doc2Vec model and the supplier and tender (from contracts) texts.
    '''
    # Extract texts from dataset
    #partitions = load_supplier_and_tender_data()
    partitions = load_test_data()
    result_partitions = []

    i = 0
    for supplier_objects, tender_objects in partitions[:1]:
        all_entities = list(supplier_objects)
        all_entities.extend(tender_objects)
        print len(all_entities)

        # Tokenize texts
        tagged_data = [TaggedDocument(words=nltk.tokenize.word_tokenize(ent.text.lower()), tags=[str(k)]) for k, ent in enumerate(all_entities)]

        # Build model
        if(shouldTrain):
            model = Doc2Vec(size=100,
                            alpha=ALPHA,
                            min_alpha=0.0025,
                            min_count=50,
                            dm=1,
                            dbow_words=0,
                            iter=2000)
            model.build_vocab(tagged_data)

            # Train
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
            model.save("models/d2v"+str(i)+"_dm100_min100.model")
            print("Model Saved")
        else:
            model = Doc2Vec.load("models/d2v"+str(i)+"_dm100_min100.model")


        # Prepare embedding outputs
        content_embeddings_data = []
        for ent in all_entities:
            vector4text = model.infer_vector(ent.text)
            content_embeddings_data.append({
                'id':ent.id,
                'name':ent.name,
                'vec':vector4text,
                'type':ent.type
            })

        if(shouldOutput):
            data = {
                        'id': [ced['id'] for ced in content_embeddings_data],
                        'name': [ced['name'] for ced in content_embeddings_data],
                        'vec': [ced['vec'] for ced in content_embeddings_data],
                        'type': [ced['type'] for ced in content_embeddings_data]
                    }
            df = pd.DataFrame(data, columns= ['id', 'name', 'vec', 'type'])
            df.to_csv('experimental_test_content_embedding_data_0' + str(i) + '.csv', index=None, header=True)

        i += 1
        result_partitions.append((model, content_embeddings_data, all_entities))
    return result_partitions


