import numpy as np
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import sys

from scipy import stats
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer, KBinsDiscretizer


from tqdm import tqdm

MIN_CORR = 0.4
N_QUERIES = 600
N_TRAIN = int(0.9 * N_QUERIES)

#normalization = sys.argv[1]

#path to group level oracle files
train_data_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/"

# trec formatted run
run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/run.train.trec"


final_train_x = []
final_train_y = []
final_val_x = []
final_val_y = []

qid2pos  = {}
with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:
    for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        if qid not in qid2pos:
            qid2pos[qid] = []
        qid2pos[qid].append(did)



qid2doc_scores = {}
with open(run, 'r') as h:
    for line in h:
        qid, _, did, pos, score, _ = line.strip().split()
        if qid not in qid2doc_scores:
            qid2doc_scores[qid] = {}
        if did not in qid2doc_scores[qid]:
            qid2doc_scores[qid][did] = [pos, score]

def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax

def process_datapoint(datapoint):

    feature_vector = np.zeros(100, dtype=float)

    qid, docids, label = datapoint.split()

    docids = docids.split(",")
    positive = docids[0] # static for single query, no need to consider
    negatives = docids[1:]

    for did in negatives:
        pos, score = qid2doc_scores[qid][did]
        feature_vector[int(pos)-1] = 1 # -1 because trec run position is 1-indexed
    
    return feature_vector, float(label)

def normalize_coefficients(coefficients, normalization):
    if normalization == "std":
        coef_mean = np.mean(coefficients)
        coef_std = np.std(coefficients)
        return (coefficients - coef_mean) / coef_std

    elif normalization == "minmax_05":
        min_coeff = np.min(coefficients)
        max_coeff = np.max(coefficients)
        normalized = (coefficients - min_coeff) / (max_coeff - min_coeff)
        return normalized / 0.5

    elif normalization == "log":
        return np.log1p(coefficients)

    elif normalization == "boxcox":
        min_coeff = np.min(coefficients)
        max_coeff = np.max(coefficients)
        normalized = (coefficients - min_coeff) / (max_coeff - min_coeff) 
        normalized = normalized / 0.1
        return stats.boxcox(normalized + 1)[0]  # Adding 1 to handle zero values

    elif normalization == "yeojohnson":
        pt = PowerTransformer(method='yeo-johnson')
        return pt.fit_transform(coefficients.reshape(-1, 1)).flatten()

    elif normalization == "quantile":
        qt = QuantileTransformer(output_distribution='normal')
        return qt.fit_transform(coefficients.reshape(-1, 1)).flatten()

    elif normalization == "robust":
        rs = RobustScaler()
        return rs.fit_transform(coefficients.reshape(-1, 1)).flatten()

    elif normalization == "binning":
        kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        return kbd.fit_transform(coefficients.reshape(-1, 1)).flatten()

    elif normalization == "log_std":
        log_coeffs = np.log1p(coefficients)
        log_mean = np.mean(log_coeffs)
        log_std = np.std(log_coeffs)
        return (log_coeffs - log_mean) / log_std

    else:
        raise ValueError(f"Unknown normalization method: {normalization}")


#for normalization in ["log", "boxcox", "yeojohnson", "quantile", "robust", "binning", "log_std"]:
for normalization in ["minmax_05"]:
    qid = None
    total = 0
    with open(f"{train_data_path}/datamodel_train_independency_{normalization}.tsv", 'w') as out_train, \
        open(f"{train_data_path}/datamodel_val_independency_{normalization}.tsv", 'w') as out_val:

        for i in tqdm(range(N_QUERIES)):
                X_train = []
                y_train = []

                with open(f"{train_data_path}/datamodel_data_q{i}.tsv", 'r') as h:
                    for line in h:
                        features, label = process_datapoint(line.strip())
                        if features is not None:
                            X_train.append(features)
                            y_train.append(label)
                    qid = line.strip().split()[0]


                X_train = np.array(X_train)
                y_train = np.array(y_train)

                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                regressor_params = {
                        'alpha': 0.01
                    }
                regressor = Ridge(random_state=42, **regressor_params)

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', regressor)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)

                #mse = mean_squared_error(y_val, y_pred)
                val_corr, _ = pearsonr(y_val, y_pred)

                ridge_regressor = pipeline.named_steps['regressor']
                coefficients = ridge_regressor.coef_
                
                standardized_coefficients = normalize_coefficients(coefficients, normalization)
                total += 1

                if val_corr >= MIN_CORR:
                    for coef, did in zip(standardized_coefficients, qid2doc_scores[qid]):
                        if did not in qid2pos[qid]:
                            if total <= N_TRAIN:
                                # final_train_x.append(f"{qid}\t{qid2pos[qid][0]}\t{did}")
                                # final_train_y.append(coef)
                                out_train.write(f"{qid}\t{qid2pos[qid][0]}\t{did}\t{coef}\n")
                            else: 
                                # final_val_x.append(f"{qid}\t{qid2pos[qid][0]}\t{did}")
                                # final_val_y.append(coef)
                                out_val.write(f"{qid}\t{qid2pos[qid][0]}\t{did}\t{coef}\n")


# concated_labels = final_train_y + final_val_y
# standardized_list = (np.array(concated_labels) - np.mean(concated_labels)) / np.std(concated_labels)

# final_train_y_std = standardized_list[:len(final_train_y)]
# final_val_y_std = standardized_list[len(final_train_y):]


# with open(f"{train_data_path}/datamodel_train_independency_std_global.tsv", 'w') as out_train, \
#     open(f"{train_data_path}/datamodel_val_independency_std_global.tsv", 'w') as out_val:

#     assert len(final_val_x) == len(final_val_y_std)
#     assert len(final_train_x) == len(final_train_y_std)

#     for x, y in zip(final_train_x, final_train_y_std):
#         out_train.write(f"{x}\t{y}\n")
    
#     for x, y in zip(final_val_x, final_val_y_std):
#         out_val.write(f"{x}\t{y}\n")

            



