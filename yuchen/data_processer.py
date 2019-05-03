import numpy as np
import os
import csv
import sys
import random
from os.path import expanduser

csv.field_size_limit(sys.maxsize)

CONTRACT_CLEAN_DATA = expanduser("~") + "/data/spend_network/input/ocds_contracts_clean.csv"
WEBSITE_SUPPLIER_DATA = expanduser("~") + "/data/spend_network/input/entity_website_text.csv"
OUTPUT_ROOT = expanduser("~") + "/data/spend_network/output/"
PLAN_A_PATH = "plan_a/"
PLAN_B_PATH = "plan_b/"
TRAINING_PATH = "training_sets/"
TESTING_PATH = "testing_sets/"
N_DUPLICATES = 10
N_FOLDS = 10
COLUMNS = ["buyer_id", "contract_id", "company_name", "contract_text", "company_text", "company_info"]

def main():
    generate_duplicates()

def generate_duplicates():
    for i in range(N_DUPLICATES):
        split_files(i)

def split_files(index_duplicate):
    with open(CONTRACT_CLEAN_DATA, "r") as csv_file:
        data_reader = csv.reader(csv_file, delimiter=",")
        next(data_reader)
        n_rows = 0
        list_rows = list()
        for row in data_reader:
            n_rows += 1
            list_rows.append(row)
        n_testing_samples = n_rows // N_FOLDS
        index_testing_samples = random.sample(list(range(n_rows)), n_testing_samples)
        list_training_samples = list()
        list_testing_samples = list()
        for index in range(n_rows):
            if index in index_testing_samples:
                list_testing_samples.append(list_rows[index])
            else:
                list_training_samples.append(list_rows[index])
        plan_a(index_duplicate, list_testing_samples, list_training_samples)
        plan_b(index_duplicate, list_testing_samples, list_training_samples)


def plan_a(index_duplicate, list_testing_samples, list_training_samples):
    # Plan A of processing data: using the home_page_text field in the
    # entity_website_text.csv file as the company_text field.
    print("Plan A - number of training samples: {}".format(len(list_training_samples)))
    print("Plan A - number of testing samples: {}".format(len(list_testing_samples)))
    dict_company_txt = dict()
    with open(WEBSITE_SUPPLIER_DATA, "r") as csv_file:
        data_reader = csv.reader(csv_file, delimiter=",")
        next(data_reader)
        for row in data_reader:
            company_name = row[0].lower()
            company_info = row[-1]
            company_text = row[2]
            if company_text == "":
                continue
            if company_name not in dict_company_txt:
                dict_company_txt[company_name] = dict()
                dict_company_txt[company_name]["info"] = company_info
                dict_company_txt[company_name]["txt"] = company_text

    training_path = "{}{}{}".format(OUTPUT_ROOT, PLAN_A_PATH, TRAINING_PATH)
    testing_path = "{}{}{}".format(OUTPUT_ROOT, PLAN_A_PATH, TESTING_PATH)
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    training_file = training_path + str(index_duplicate) + ".csv"
    testing_file = testing_path + str(index_duplicate) + ".csv"

    debug_n_training_hit = 0
    debug_n_training_miss = 0
    debug_n_testing_hit = 0
    debug_n_testing_miss = 0

    with open(training_file, "w", newline="") as csv_training_file:
        file_writer = csv.writer(csv_training_file, delimiter=",")
        file_writer.writerow(COLUMNS)
        for row in list_training_samples:
            contract_id = row[1] # ocid
            buyer_id = row[11] # the id of the buyer of the contact
            company_name = row[14] # the name of the supplier of the contract,
                                # also used as an index in the entity_website_text.csv file
            contract_text = row[4] + ". " + row[5] + ". " + row[7] + ". " + row[8] + ". " + row[9] + ". " + row[10]
            if company_name.lower() not in dict_company_txt:
                debug_n_training_miss += 1
                continue
            else:
                company_name = company_name.lower()
                company_text = dict_company_txt[company_name]["txt"]
                company_info = dict_company_txt[company_name]["info"]
                file_writer.writerow([buyer_id, contract_id, company_name,
                    contract_text, company_text, company_info])
                debug_n_training_hit += 1

    with open(testing_file, "w", newline="") as csv_testing_file:
        file_writer = csv.writer(csv_testing_file, delimiter=",")
        file_writer.writerow(COLUMNS)
        for row in list_testing_samples:
            contract_id = row[1] # ocid
            buyer_id = row[11] # the id of the buyer of the contact
            company_name = row[14] # the name of the supplier of the contract,
                                # also used as an index in the entity_website_text.csv file
            contract_text = row[4] + ". " + row[5] + ". " + row[7] + ". " + row[8] + ". " + row[9] + ". " + row[10]
            if company_name.lower() not in dict_company_txt:
                debug_n_testing_miss += 1
                continue
            else:
                company_name = company_name.lower()
                company_text = dict_company_txt[company_name]["txt"]
                company_info = dict_company_txt[company_name]["info"]
                file_writer.writerow([buyer_id, contract_id, company_name,
                    contract_text, company_text, company_info])
                debug_n_testing_hit += 1

def plan_b(index_duplicate, list_testing_samples, list_training_samples):
    # Plan B of processing data: to generate the text of each company
    # (supplier), find the texts of all the contracts that have been awarded by
    # the company and concatenate these texts together.
    print("Plan B - number of training samples: {}".format(len(list_training_samples)))
    print("Plan B - number of testing samples: {}".format(len(list_testing_samples)))
    dict_company_txt = dict()
    for row in list_training_samples:
        company_name = row[14].lower()
        company_text = row[-1]
        if company_name not in dict_company_txt:
            dict_company_txt[company_name] = company_text
        else:
            dict_company_txt[company_name] += company_text

    training_path = "{}{}{}".format(OUTPUT_ROOT, PLAN_B_PATH, TRAINING_PATH)
    testing_path = "{}{}{}".format(OUTPUT_ROOT, PLAN_B_PATH, TESTING_PATH)
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    training_file = training_path + str(index_duplicate) + ".csv"
    testing_file = testing_path + str(index_duplicate) + ".csv"

    with open(training_file, "w", newline="") as csv_training_file:
        file_writer = csv.writer(csv_training_file, delimiter=",")
        file_writer.writerow(COLUMNS)
        for row in list_training_samples:
            contract_id = row[1] # ocid
            buyer_id = row[11] # the id of the buyer of the contact
            company_name = row[14] # the name of the supplier of the contract,
                                # also used as an index in the entity_website_text.csv file
            company_name = company_name.lower()
            contract_text = row[4] + ". " + row[5] + ". " + row[7] + ". " + row[8] + ". " + row[9] + ". " + row[10]
            company_info = ""
            company_text = dict_company_txt[company_name]
            file_writer.writerow([buyer_id, contract_id, company_name,
                contract_text, company_text, company_info])

    with open(testing_file, "w", newline="") as csv_testing_file:
        file_writer = csv.writer(csv_testing_file, delimiter=",")
        file_writer.writerow(COLUMNS)
        for row in list_testing_samples:
            contract_id = row[1] # ocid
            buyer_id = row[11] # the id of the buyer of the contact
            company_name = row[14] # the name of the supplier of the contract,
                                # also used as an index in the entity_website_text.csv file

            company_name = company_name.lower()
            contract_text = row[4] + ". " + row[5] + ". " + row[7] + ". " + row[8] + ". " + row[9] + ". " + row[10]
            if company_name not in dict_company_txt:
                # company_text = ""
                continue
            else:
                company_text = dict_company_txt[company_name]
            file_writer.writerow([buyer_id, contract_id, company_name,
                contract_text, company_text, company_info])

if __name__ == "__main__":
    main()
