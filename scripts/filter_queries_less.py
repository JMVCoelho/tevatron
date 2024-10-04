import pandas as pd

def process_tsv(df, top_percent):
    # Assuming the last column contains the scores
    score_column_index = df.shape[1] - 1  # Last column index

    # Sort the DataFrame by the score column in descending order
    sorted_df = df.sort_values(by=score_column_index, ascending=False)

    # Calculate the number of top lines to take
    num_top_lines = int(len(sorted_df) * (top_percent / 100))

    # Select the top X% lines
    top_lines = sorted_df.head(num_top_lines)

    return top_lines

def split_dataframe_by_score(df):
    even_indices = df.index[::2]
    odd_indices = df.index[1::2]

    # Extract the even and odd rows
    even_rows = df.iloc[even_indices]
    odd_rows = df.iloc[odd_indices]

    # Compare the scores (assuming the last column contains the scores)
    mask = even_rows.iloc[:, -1].values >= odd_rows.iloc[:, -1].values

    # Apply the mask to assign rows to df1 and df2
    higher = pd.concat([even_rows[mask], odd_rows[~mask]]).sort_index()
    lower = pd.concat([odd_rows[mask], even_rows[~mask]]).sort_index()

    #assert higher.iloc[:, 2].tolist() == lower.iloc[:, 2].tolist() # one q per doc!!!

    return higher, lower


model = "pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu"

gen = "gen10"

file_path = f'/data/user_data/jmcoelho/embeddings/marco_docs/{model}/{gen}_less_influence_v3/run_all_queries_with_valid_grad_dotp'

PAIRS = True

for perc in [5]:

    if not PAIRS:
        df = pd.read_csv(file_path, sep='\t', header=None)
        worst = None
    else:
        df = pd.read_csv(file_path, sep='\t', header=None)

        best, worst =  split_dataframe_by_score(df)

        print(best.head(10))
        print(worst.head(10))
        df = best

        
    top_lines = process_tsv(df, perc).sort_values(0)[[0, 1]]
    top_lines.to_csv(f"/data/user_data/jmcoelho/embeddings/marco_docs/{model}/{gen}-shnegs/queries.random.momentum.shn.top100.{perc}.less.filter.txt", sep='\t', index=False, header=False)
        