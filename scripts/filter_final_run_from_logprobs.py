import pandas as pd
import numpy as np


def softmax(x, temperature=1.0):
    """Compute the softmax of vector x scaled by temperature."""
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))  # for numerical stability
    return e_x / e_x.sum()

def keep_top_l_lines(tsv_file, L, output_file):
    df = pd.read_csv(tsv_file, sep='\t', names=['qid', 'q0', 'did', 'rel', 'score'])
    df_sorted = df.sort_values(by='score', ascending=False)
    df_top_l = df_sorted.head(L)
    df_top_l = df_top_l.drop(columns=['score'])
    df_top_l.to_csv(output_file, sep='\t', index=False, header=False)
    return df_top_l

def keep_random_lines(tsv_file, L, output_file):
    df = pd.read_csv(tsv_file, sep='\t', names=['qid', 'q0', 'did', 'rel', 'score'])    
    df_random_l = df.sample(n=L, random_state=17121998, replace=False)
    df_random_l = df_random_l.drop(columns=['score'])
    df_random_l.to_csv(output_file, sep='\t', index=False, header=False)
    return df_random_l

def sample_lines(tsv_file, L, output_file, temperature=1.0):
    df = pd.read_csv(tsv_file, sep='\t', names=['qid', 'q0', 'did', 'rel', 'score'])
    probabilities = softmax(df['score'].values, temperature)    
    df_sampled = df.sample(n=L, weights=probabilities, random_state=17121998, replace=False)    
    df_sampled = df_sampled.drop(columns=['score'])
    df_sampled.to_csv(output_file, sep='\t', index=False, header=False)
    return df_sampled

def filter_by_qids(filtered_qrels_df, tsv_file_2, output_file_2):    
    qids_to_keep = set(filtered_qrels_df['qid'].unique())
    df_text = pd.read_csv(tsv_file_2, sep='\t', names=['qid', 'text'])
    df_filtered_text = df_text[df_text['qid'].isin(qids_to_keep)]
    df_filtered_text.to_csv(output_file_2, sep='\t', index=False, header=False)



all_qrels = '/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen17.rr.tsv'
filtered_qrels = '/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen17.rr.sample.tsv'
filtered_qrels_df = keep_top_l_lines(all_qrels, 360000, filtered_qrels)

all_queries =  '/data/user_data/jmcoelho/datasets/marco/documents/gen17.query.tsv'
filtered_queries = '/data/user_data/jmcoelho/datasets/marco/documents/gen17.query.rr.sample.tsv'
filter_by_qids(filtered_qrels_df, all_queries, filtered_queries)