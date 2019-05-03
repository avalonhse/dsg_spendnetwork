This folder contains the training data sets and testing data sets of the Spend
Network Challenge.

There are two folders that contain the data generated through two different
ways. In plan_a/, the text of a company is generated from the texts in the
entity_website_text.csv. In plan_b/ the text of a company is generated from
the texts of all the contracts that were awarded by the company. The texts of
the contracts are simply concatenated.

In both plan_a/ and plan_b/, there are two folders (training_sets/ and
testing_sets/) containing the training data sets and the testing data sets.
There are multiple duplicates (currently 10) for both data sets and the files of
each duplicate are named as number.csv files. Each training-testing pair was
generated through 10-fold cross validation (1 for testing and 9 for training).

The fields in each file include:
    * buyer_id: the id of the buyer of the contract
    * contract_id: the id of the contract
    * company_name: the name of the company of the contract
    * contract_text: the text related to the contract
    * company_text: the text related to the company (different in plan_a and
      plan_b)
    * company_info: the "about us and contact us" information of the company
      (extracted from data table 6), only valid for plan_a

NOTE: In the testing sets of plan_b, some company texts are "". This is because
these companies do not appear in the training sets. This is similar to the
cold-start problem of recommender systems (new user without any information). 
