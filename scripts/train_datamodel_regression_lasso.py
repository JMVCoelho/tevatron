import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import optuna

train_data_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_single_query_2k/datamodel_train.tsv"
val_data_path =  "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_single_query_2k/datamodel_val.tsv"
test_data_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_single_query_2k/datamodel_test.tsv"


run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/run.train.trec"

qid2doc_scores = {}
with open(run, 'r') as h:
    for line in h:
        qid, _, did, pos, score, _ = line.strip().split()
        if qid not in qid2doc_scores:
            qid2doc_scores[qid] = {}
        if did not in qid2doc_scores[qid]:
            qid2doc_scores[qid][did] = [pos, score]


def process_datapoint(datapoint):

    feature_vector = np.zeros(100, dtype=float)

    qid, docids, label = datapoint.split()

    docids = docids.split(",")
    positive = docids[0]
    negatives = docids[1:]

    # if positive not in qid2doc_scores[qid]:
    #     return None, None
    # _, score = qid2doc_scores[qid][positive]
    # feature_vector[0] = float(score)

    # for did in negatives:
    #     pos, score = qid2doc_scores[qid][did]
    #     feature_vector[int(pos)] = float(score)
    
    # return feature_vector, float(label)
    for did in negatives:
        if did not in qid2doc_scores[qid]:
            print("tough luck")
            return None, None
        pos, score = qid2doc_scores[qid][did]
        feature_vector[int(pos)-1] = score
    
    return feature_vector, float(label)


X_train = []
y_train = []

with open(train_data_path, 'r') as h:
    for line in h:
        features, label = process_datapoint(line.strip())
        if features is not None:
            X_train.append(features)
            y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = []
y_val = []

with open(val_data_path, 'r') as h:
    for line in h:
        features, label = process_datapoint(line.strip())
        if features is not None:
            X_val.append(features)
            y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

def objective_mlp(trial):
    
    # Define MLPRegressor with parameters to optimize
    regressor_params = {
        'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', 50, 200, log=True),
        'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
    }
    regressor = MLPRegressor(random_state=42, **regressor_params)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Predict on validation data
    y_pred = pipeline.predict(X_val)
    
    # Calculate mean squared error (you can use other metrics as needed)
    mse = mean_squared_error(y_val, y_pred)
    val_corr, _ = pearsonr(y_val, y_pred)
    
    return mse, val_corr

def objective_lasso(trial):
    
    # Define MLPRegressor with parameters to optimize
    regressor_params = {
        'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=False),
    }
    regressor = Ridge(random_state=42, **regressor_params)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Predict on validation data
    y_pred = pipeline.predict(X_val)
    
    # Calculate mean squared error (you can use other metrics as needed)
    mse = mean_squared_error(y_val, y_pred)
    val_corr, _ = pearsonr(y_val, y_pred)
    
    return mse, val_corr


study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective_mlp, n_trials=20)

print(study.best_trials)

optuna.visualization.plot_pareto_front(study, target_names=["MSE", "CORRELATION"])


# regressor_params = {
#         'alpha': 0.01
#     }
# regressor = Ridge(random_state=42, **regressor_params)

# # Create pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('regressor', regressor)
# ])

# # Fit the pipeline on training data
# pipeline.fit(X_train, y_train)

# # Predict on validation data
# y_pred = pipeline.predict(X_val)

# # Calculate mean squared error (you can use other metrics as needed)
# mse = mean_squared_error(y_val, y_pred)
# val_corr, _ = pearsonr(y_val, y_pred)


# print(f"MSE: {mse}")
# print(f"CORR: {val_corr}")
# # Accessing the coefficients from the Ridge regressor
# ridge_regressor = pipeline.named_steps['regressor']
# coefficients = ridge_regressor.coef_

# # Printing the coefficients
# print("Coefficients of the linear model:")
# print(coefficients)
