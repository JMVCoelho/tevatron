import pandas as pd

# Load the TSV file
# df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level/run_all_queries_lower_loss', sep='\t', header=None, names=['qid', 'negs', 'g', 'val_loss', 'grad_norm'])

# # Sort the DataFrame by 'col3' in descending order
# df_sorted = df.sort_values(by='grad_norm', ascending=False)

# half_index = len(df_sorted) // 2
# pc75_index = int(len(df_sorted) * 0.75)

# df_top_half = df_sorted.iloc[:half_index].drop(columns=['g', 'val_loss', 'grad_norm'])

# df_bottom_half = df_sorted.iloc[half_index:].drop(columns=['g', 'val_loss', 'grad_norm'])


# #df_top_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_mates_sample_level/higher_loss_filtered', sep='\t', header=False, index=False)
# df_bottom_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level/run_all_queries_lower_loss_filtered_v2', sep='\t', header=False, index=False)

# #########


# import pandas as pd

# df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates/run_all_queries_with_valid_loss_filtered', sep='\t', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5'])

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
#     top_half, bottom_half = split_group(group)
#     top_half_list.append(top_half)
#     bottom_half_list.append(bottom_half)



# # Concatenate all the top halves and bottom halves
# df_top_half = pd.concat(top_half_list).drop(columns=['col3', 'col4', 'col5'])
# df_bottom_half = pd.concat(bottom_half_list).drop(columns=['col3', 'col4', 'col5'])

# # Save the two halves to separate TSV files without headers and without 'col4'
# df_top_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates/higher_loss_filtered_mates_worst', sep='\t', header=False, index=False)
# df_bottom_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates/lower_loss_filtered_mates_best', sep='\t', header=False, index=False)

###### 

# import pandas as pd

# df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level/run_all_queries_with_valid_loss', sep='\t', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5'])

# df = df.drop(columns=['col1', 'col2', 'col3'])
# pearson_corr = df['col4'].corr(df['col5'], method='pearson')

# # Spearman correlation
# spearman_corr = df['col4'].corr(df['col5'], method='spearman')

# print(pearson_corr, spearman_corr)
# exit()
# df = df[df["col3"]==1].drop(columns=['col2', 'col3', 'col5'])
# #df['col4'] = (df['col4'] - df['col4'].mean()) / df['col4'].std()
# df['col4'] = 2 * (df['col4'] - df['col4'].min()) / (df['col4'].max() - df['col4'].min())


# df.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v4/datamodel_full_minmax', sep='\t', header=False, index=False)
# # # Save the two halves to separate TSV files without headers and without 'col4'
# # df_top_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates/higher_loss_filtered_mates_worst', sep='\t', header=False, index=False)
# # df_bottom_half.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates/lower_loss_filtered_mates_best', sep='\t', header=False, index=False)

# ##### 


# import pandas as pd

# # Load the main data file
# df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/run_all_queries_grouped_with_valid', sep='\t', header=None, names=['col1', 'col2', 'col3', 'col4'])

# # Load the initial values file
# initial_values_df = pd.read_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/run_split_0_initial_valid_loss', sep='\t', header=None, names=['col3', 'initial_value'])

# # Merge the initial values with the main DataFrame on 'col3'
# df = df.merge(initial_values_df, on='col3')

# # Filter rows where col4 is smaller than the corresponding initial_value
# filtered_df = df[df['col4'] > df['initial_value']]

# # Drop the columns that are not needed in the output
# filtered_df = filtered_df.drop(columns=['col3', 'col4', 'initial_value'])

# # Save the filtered DataFrame to a TSV file without headers and index
# filtered_df.to_csv('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped/hurting_only', sep='\t', header=False, index=False)


####

# mapper = {}
# with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_less_sample_level/run_all_queries_with_valid_grad_dot_p", 'r') as h:
#     for line in h:
#         qid, negs, less_score = line.strip().split("\t")

#         if qid not in mapper:
#             mapper[qid] = [negs, less_score]
#         else:
#             print("err???")


# with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_less_sample_level/higher_less", 'w') as out1, \
#     open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_less_sample_level/lower_less", 'w') as out2:
#     for num in range(0, 49999, 2):
#         q0 = str(num)
#         q1 = str(num + 1)

#         try:
#             contender_0 = mapper[q0]
#             contender_1 = mapper[q1]
#         except Exception:
#             continue

#         if contender_0[1] > contender_1[1]:
#             out1.write(f"{q0}\t{contender_0[0]}\n")
#             out2.write(f"{q1}\t{contender_1[0]}\n")

#         else:
#             out1.write(f"{q1}\t{contender_1[0]}\n")
#             out2.write(f"{q0}\t{contender_0[0]}\n")



import csv

# Function to process the TSV file
def process_tsv(input_file, output_file, initial_group_loss, pick_min, row_n):
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t')
        rows = list(reader)

    group_to_loss = {}

    with open(initial_group_loss, 'r') as h:
        for line in h:
            group, initial_loss = line.strip().split("\t")
            group_to_loss[group] = initial_loss

    # List to store the filtered rows
    filtered_rows = []

    # Process every group of 4 rows
    for i in range(0, len(rows), 4):
        group = rows[i:i+4]
        
        # Find the row with the lowest value in the fourth column

        group_value = group[0][2]

        assert all(row[2] == group_value for row in group)

        if any(float(row[row_n]) < float(group_to_loss[group_value]) for row in group):

            if pick_min:
                min_row = min(group, key=lambda row: float(row[row_n]))
                filtered_rows.append(min_row)
            else:
                max_row = max(group, key=lambda row: float(row[row_n]))
                filtered_rows.append(max_row)
    
    # Write the filtered rows to a new TSV file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerows(filtered_rows)

# Example usage
input_file = '/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v3/run_all_queries_with_valid_loss'
output_file = '/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v3/lower_loss_filtered'
initial_valid_loss = '/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v3/run_split_0_initial_valid_loss'
process_tsv(input_file, output_file, initial_valid_loss, pick_min=True, row_n=3)