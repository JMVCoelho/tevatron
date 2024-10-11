import pandas as pd
import sys

# model = "pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision"
# gen = "gen11"


model = sys.argv[1]
gen = sys.argv[2]
percentage = int(sys.argv[3])

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
    grouped = df.groupby(df.columns[2])
    highest = grouped.apply(lambda x: x.loc[x.iloc[:, 4].idxmax()])
    lowest = grouped.apply(lambda x: x.loc[x.iloc[:, 4].idxmin()])
    
    highest = highest.reset_index(drop=True)
    lowest = lowest.reset_index(drop=True)
    
    return highest, lowest




file_path = f'/data/user_data/jmcoelho/embeddings/marco_docs/{model}/{gen}-shnegs/queries.random.shn.top100.txt'

for perc in [percentage]:

    df = pd.read_csv(f'/data/user_data/jmcoelho/datasets/marco/documents/qrels.{gen}.rr.tsv', sep='\t', header=None)
    print(df.head(10))
    best, worst =  split_dataframe_by_score(df)
    print("#######")
    print(best.head(10))
    print(worst.head(10))

    df = best
    
    top_lines = process_tsv(df, perc).sort_values(by=2)

    if worst is not None:
        subset_worst = worst[worst[2].isin(top_lines[2])].sort_values(by=2)

        subset_worst = subset_worst[[2,0]]
        top_lines = top_lines[[2,0]]

        merged_df = pd.merge(top_lines, subset_worst, left_on=2, right_on=2)
        merged_df.columns = ['prompt', 'chosen', 'rejected']

        merged_df.to_csv(f'/data/user_data/jmcoelho/embeddings/marco_docs/{model}/{gen}-shnegs/dpo-data-ids-{perc}.tsv', sep='\t', index=False)


    qids = set(top_lines[0].tolist())

    with open(f"/data/user_data/jmcoelho/datasets/marco/documents/{gen}.query.{perc}.subset.tsv", 'w') as out:
        with open(f"/data/user_data/jmcoelho/datasets/marco/documents/{gen}.query.tsv", 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                if int(qid) in qids:
                    out.write(f"{qid}\t{text}\n")

    with open(f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.{gen}.{perc}.subset.tsv", 'w') as out:
        with open(f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.{gen}.tsv", 'r') as h:
            for line in h:
                qid, q0, did, rel = line.strip().split("\t")
                if int(qid) in qids:
                    out.write(f"{qid}\t{q0}\t{did}\t{rel}\n")
