import networkx as nx
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


def network_analysis_matching():

    # dataframe of distance measures in text matching
    filename = 'distance0'
    div_df = pd.read_csv('/Users/erinclark/Downloads/%s.csv' % filename, index_col=0)

    div_df = div_df.transpose()
    maxdist = div_df.values.max()

    print 'Filename: %s' % filename
    print 'Maximum overall dist: %s' % maxdist
    # use this if we need to scale it between 0-1
    # scaled_div_df = div_df.apply(lambda x: x/maxdist)

    # real matches of OCIDs to supplier names
    full_df = pd.read_csv('/Users/erinclark/git/AlanTuringInst/SpendNetworkChallenge/ocds_contracts_clean.csv')
    real_match_df = full_df[['ocid', 'supplier']]
    real_match_df = real_match_df.set_index('ocid')
    real_match_df['supplier'] = real_match_df['supplier'].str.lower()

    # dataframe to hold bool of true ocid to supplier matches
    adj_real_df = pd.DataFrame(columns=div_df.columns, index=div_df.index)
    adj_real_df[:] = 0

    dist_true_array = []
    for index, row in adj_real_df.iterrows():

        supplier_name = real_match_df.loc[index, 'supplier']
        ocid = index
        adj_real_df.loc[ocid, supplier_name] = 1
        dist_true_array.append(div_df.loc[ocid, supplier_name])

    print 'Highest true match distance: %s' % max(dist_true_array)
    print 'Mean %s' % np.mean(dist_true_array)
    print 'Median %s' % np.median(dist_true_array)
    print 'Number of contracts: %s' % len(dist_true_array)
    # for i in sorted(dist_true_array):
    #     print i

    threshold = np.median(dist_true_array)

    # dataframe to hold bool of MATCHED ocid to supplier matches aka predicted tenders for a supplier
    adj_matched_df = pd.DataFrame(columns=div_df.columns, index=div_df.index)
    adj_matched_df[:] = 0

    threshold_vs_retreval_df = pd.DataFrame(columns=('threshold', 'retrieval'))
    retrieval_by_supplier_df = pd.DataFrame()
    for threshold in np.arange(0, maxdist + 0.1, 0.1):

        for index, row in div_df.iterrows():
            adj_matched_df.loc[index, :] = row.apply(lambda x: 1 if x <= threshold else 0)

        accuracy_df = pd.DataFrame(columns=div_df.columns, index=div_df.index)
        for index, row in adj_matched_df.iterrows():

            supplier_name = real_match_df.loc[index, 'supplier']
            ocid = index
            if adj_matched_df.loc[ocid, supplier_name] == 1:
                result = True
            else:
                result = False
            accuracy_df.loc[ocid, supplier_name] = result

        retrivalvals = []
        ret_by_sup_row = {'threshold': threshold}
        for sup_name in accuracy_df.columns:

            value_series = accuracy_df[sup_name].value_counts()
            try:
                truecount = value_series[True]
            except KeyError:
                truecount = 0

            total_match = accuracy_df[sup_name].value_counts().sum()
            falsecount = total_match - truecount

            tot_above_threshold = div_df[div_df[sup_name] > threshold][sup_name].count()
            tot_below_threshold = div_df[div_df[sup_name] <= threshold][sup_name].count()

            # use when plotting suppliers separately
            # if total_match < 3:
            #     continue

            scaled_test = (1 / (1 + np.exp(falsecount - truecount)))
            retrivalvals.append(scaled_test)

            ret_by_sup_row.update({sup_name: scaled_test})
            # use this to give cont counts per supplier
            # ret_by_sup_row.update({'(%s) %s' % (str(total_match), sup_name): scaled_test})

            # print 'true: %s' % truecount
            # print 'False: %s' % falsecount
            # print 'total: %s' % total_match
            # print 'Score: %s' % scaled_test

        # print 'Highest retreval %s' % max(retrivalvals)
        # print 'Mean %s' % np.mean(retrivalvals)
        # print 'Median %s' % np.median(retrivalvals)
        fmean = np.mean(retrivalvals)

        dict = {
            'threshold': threshold,
            'retrieval': fmean
        }
        threshold_vs_retreval_df = threshold_vs_retreval_df.append(dict,  ignore_index=True)
        retrieval_by_supplier_df = retrieval_by_supplier_df.append(ret_by_sup_row, ignore_index=True)

    print retrieval_by_supplier_df
    palette = plt.get_cmap('Set1')
    num = 0
    columns = []
    # for multiple lines on same graph:
    # for column in retrieval_by_supplier_df.drop('threshold', axis=1):
    #     columns.append(column)
    #     num += 1
    #     plt.plot(retrieval_by_supplier_df['threshold'], retrieval_by_supplier_df[column], linewidth=1, color=palette(num), label=column)    #, alpha=0.9, label=column)


    plt.plot(threshold_vs_retreval_df['threshold'], threshold_vs_retreval_df['retrieval'], linewidth=1)
    # plt.scatter(threshold_vs_retreval_df['threshold'], threshold_vs_retreval_df['retrieval'])

    plt.xlabel('Threshold of distance measure')
    plt.ylabel('Retrieval score')
    # plt.grid()
    plt.legend(loc='best', fontsize='xx-small')
    plt.show()



if __name__ == '__main__':

    network_analysis_matching()
