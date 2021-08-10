# import required modules
import os
import pandas as pd

# import data from previous script
os.getcwd()
os.chdir(r"D:\Libraries\Documents\Data Science\JCU_MDS\2021_Masterclass 1\Assignment 2")

content_train_metrics = pd.read_csv('Part 3_content train metrics.csv', index_col=0).sort_values(by='USER_ID')
content_test_metrics = pd.read_csv('Part 3_content test metrics.csv', index_col=0).sort_values(by='USER_ID')
collab_train_metrics = pd.read_csv('Part 3_collab train metrics.csv', index_col=0).sort_values(by='USER_ID')
collab_test_metrics = pd.read_csv('Part 3_collab test metrics.csv', index_col=0).sort_values(by='USER_ID')

# calculate average precision, recall, F1 and similarity scores, and average number of titles recommended already in
#  the course list of that particular user
content_train_MAPK = content_train_metrics['PRECISION@K'].mean()
content_train_MARK = content_train_metrics['RECALL@K'].mean()
content_train_MF1K = content_train_metrics['F1@K'].mean()
content_train_MASS = content_train_metrics['AVG_SIM_SCORE@K'].mean()
content_train_ACR = content_train_metrics['NUM_CORRECT_RECS'].mean()

content_test_MAPK = content_test_metrics['PRECISION@K'].mean()
content_test_MARK = content_test_metrics['RECALL@K'].mean()
content_test_MF1K = content_test_metrics['F1@K'].mean()
content_test_MASS = content_test_metrics['AVG_SIM_SCORE@K'].mean()
content_test_ACR = content_test_metrics['NUM_CORRECT_RECS'].mean()

collab_train_MAPK = collab_train_metrics['PRECISION@K'].mean()
collab_train_MARK = collab_train_metrics['RECALL@K'].mean()
collab_train_MF1K = collab_train_metrics['F1@K'].mean()
collab_train_MASS = collab_train_metrics['AVG_SIM_SCORE@K'].mean()
collab_train_ACR = collab_train_metrics['NUM_CORRECT_RECS'].mean()

collab_test_MAPK = collab_test_metrics['PRECISION@K'].mean()
collab_test_MARK = collab_test_metrics['RECALL@K'].mean()
collab_test_MF1K = collab_test_metrics['F1@K'].mean()
collab_test_MASS = collab_test_metrics['AVG_SIM_SCORE@K'].mean()
collab_test_ACR = collab_test_metrics['NUM_CORRECT_RECS'].mean()

content_train_summary_stats = \
    content_train_metrics[['NUM_RELEVANT_ITEMS', 'NUM_CORRECT_RECS', 'AVG_SIM_SCORE@K',
                           'PRECISION@K', 'RECALL@K', 'F1@K']].describe()
content_test_summary_stats = \
    content_test_metrics[['NUM_RELEVANT_ITEMS', 'NUM_CORRECT_RECS', 'AVG_SIM_SCORE@K',
                          'PRECISION@K', 'RECALL@K', 'F1@K']].describe()

collab_train_summary_stats = \
    collab_train_metrics[['NUM_RELEVANT_ITEMS', 'NUM_CORRECT_RECS', 'AVG_SIM_SCORE@K',
                          'PRECISION@K', 'RECALL@K', 'F1@K']].describe()
collab_test_summary_stats = \
    collab_test_metrics[['NUM_RELEVANT_ITEMS', 'NUM_CORRECT_RECS', 'AVG_SIM_SCORE@K',
                         'PRECISION@K', 'RECALL@K', 'F1@K']].describe()

content_train_summary_stats.to_csv("Part 4_content_train_summary_stats.csv")
content_test_summary_stats.to_csv("Part 4_content_test_summary_stats.csv")
collab_train_summary_stats.to_csv("Part 4_collab_train_summary_stats.csv")
collab_test_summary_stats.to_csv("Part 4_collab_test_summary_stats.csv")



