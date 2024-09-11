import pandas as pd

# Load the TSV file
df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped_gradvar/run_all_queries_with_grad_var', sep='\t', header=None, names=['qid', 'negs', 'val_loss'])

# Sort the DataFrame by 'col3' in descending order
df_sorted = df.sort_values(by='val_loss', ascending=False)

half_index = len(df_sorted) // 2

df_top_half = df_sorted.iloc[:half_index].drop(columns=['val_loss'])

df_bottom_half = df_sorted.iloc[half_index:].drop(columns=['val_loss'])

df_top_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped_gradvar/higher_grad_var', sep='\t', header=False, index=False)
df_bottom_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped_gradvar/lower_grad_var', sep='\t', header=False, index=False)


# import pandas as pd

# df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/run_all_queries_grouped_with_valid', sep='\t', header=None, names=['col1', 'col2', 'col3', 'col4'])

# def split_group(group):
#     group_sorted = group.sort_values(by='col4', ascending=False)
#     half_index = len(group_sorted) // 2
#     return group_sorted.iloc[:half_index], group_sorted.iloc[half_index:]

# def split_group_dyn(group, pc=0.1):
#     group_sorted = group.sort_values(by='col4', ascending=False)

#     index = int(len(group_sorted) * pc)
#     top = group_sorted.head(index)
#     bottom = group_sorted.tail(index)
#     return top, bottom

# top_half_list = []
# bottom_half_list = []

# for _, group in df.groupby('col3'):
#     top_half, bottom_half = split_group_dyn(group)
#     top_half_list.append(top_half)
#     bottom_half_list.append(bottom_half)

# # Concatenate all the top halves and bottom halves
# df_top_half = pd.concat(top_half_list).drop(columns=['col3', 'col4'])
# df_bottom_half = pd.concat(bottom_half_list).drop(columns=['col3', 'col4'])

# # Save the two halves to separate TSV files without headers and without 'col4'
# df_top_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/higher_loss_pc', sep='\t', header=False, index=False)
# df_bottom_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/lower_loss_pc', sep='\t', header=False, index=False)