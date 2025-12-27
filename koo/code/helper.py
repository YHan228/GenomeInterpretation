import os, sys
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
import torch
from model_zoo import torch_models



def load_data(file_path, reverse_compliment=False):

    # load dataset
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.float32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_synthetic_models(filepath, dataset='test'):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')
    if dataset == 'train':
        return np.array(trainmat['model_train']).astype(np.float32)
    elif dataset == 'valid':
        return np.array(trainmat['model_valid']).astype(np.float32)
    elif dataset == 'test':
        return np.array(trainmat['model_test']).astype(np.float32)


def load_basset_dataset(filepath, reverse_compliment=False):

    trainmat = h5py.File(filepath, 'r')

    x_train = np.array(trainmat['train_in']).astype(np.float32)
    y_train = np.array(trainmat['train_out']).astype(np.int32)
    x_valid = np.array(trainmat['valid_in']).astype(np.float32)
    y_valid = np.array(trainmat['valid_out']).astype(np.int32)
    x_test = np.array(trainmat['test_in']).astype(np.float32)
    y_test = np.array(trainmat['test_out']).astype(np.int32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])


    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test
    

def load_model(model_name, activation='relu', last_layer_mode='flat', input_shape=200):

    if model_name == 'cnn-50':
        from model_zoo import cnn_model
        model = cnn_model.model([50, 2], activation, input_shape)

    elif model_name == 'cnn-2':
        from model_zoo import cnn_model
        model = cnn_model.model([2, 50], activation, input_shape)

    elif model_name == 'cnn-deep':
        from model_zoo import cnn_deep
        model = cnn_deep.model(activation, input_shape)

    elif model_name == 'cnn-local':
        model = torch_models.CnnLocal(activation, last_layer_mode)

    elif model_name == 'cnn-dist':
        model = torch_models.CnnDist(activation, last_layer_mode)

    elif model_name == 'cnn-local-deep':
        # Repo version (mislabeled as cnn-dist): k=19 first layer, so actually a deeper cnn-local
        model = torch_models.CnnLocalDeep(activation, last_layer_mode)

    elif model_name == 'basset':
        from model_zoo import basset
        model = basset.model(activation)

    elif model_name == 'residualbind':
        from model_zoo import residualbind
        model = residualbind.model(activation)

    return model



def match_hits_to_ground_truth(file_path, motifs, size=32):
    
    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')
    
    # loop through filters
    best_qvalues = np.ones(size)
    best_match = np.zeros(size)
    correction = 0  
    for name in np.unique(df['Query_ID'][:-3].to_numpy()):
        filter_index = int(name.split('r')[1])

        # get tomtom hits for filter
        subdf = df.loc[df['Query_ID'] == name]
        targets = subdf['Target_ID'].to_numpy()

        # loop through ground truth motifs
        for k, motif in enumerate(motifs): 

            # loop through variations of ground truth motif
            for motifid in motif: 

                # check if there is a match
                index = np.where((targets == motifid) ==  True)[0]
                if len(index) > 0:
                    qvalue = subdf['q-value'].to_numpy()[index]

                    # check to see if better motif hit, if so, update
                    if best_qvalues[filter_index] > qvalue:
                        best_qvalues[filter_index] = qvalue
                        best_match[filter_index] = k 

        index = np.where((targets == 'MA0615.1') ==  True)[0]
        if len(index) > 0:
            if len(targets) == 1:
                correction += 1

    # get the minimum q-value for each motif
    num_motifs = len(motifs)
    min_qvalue = np.zeros(num_motifs)
    for i in range(num_motifs):
        index = np.where(best_match == i)[0]
        if len(index) > 0:
            min_qvalue[i] = np.min(best_qvalues[index])

    match_index = np.where(best_qvalues != 1)[0]
    if any(match_index):
        match_fraction = len(match_index)/float(size)
    else:
        match_fraction = 0
    
    num_matches = len(np.unique(df['Query_ID']))-3
    match_any = (num_matches - correction)/size

    return best_qvalues, best_match, min_qvalue, match_fraction, match_any



        
def interpretability_performance(X, score, X_model, precomputed_labels=None):
    """
    Calculate interpretability performance (AUROC, AUPR) for attribution scores.

    Args:
        X: Input sequences (N, L, A)
        score: Attribution scores (N, L, A)
        X_model: Ground truth model (N, A, L)
        precomputed_labels: Optional pre-computed labels (N, L) to avoid recomputation

    Returns:
        roc_score: Array of AUROC values per sample
        pr_score: Array of AUPR values per sample
    """
    score = np.abs(np.sum(score, axis=2))  # abs() to capture importance regardless of direction

    # Pre-compute labels if not provided
    if precomputed_labels is None:
        gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)  # (N, L)
        labels = (gt_info > 0.01).astype(np.float32)
    else:
        labels = precomputed_labels

    pr_score = []
    roc_score = []
    for j, gs in enumerate(score):
        label = labels[j]

        # precision recall metric
        precision, recall, thresholds = precision_recall_curve(label, gs)
        pr_score.append(auc(recall, precision))

        # roc curve
        fpr, tpr, thresholds = roc_curve(label, gs)
        roc_score.append(auc(fpr, tpr))

    roc_score = np.array(roc_score)
    pr_score = np.array(pr_score)

    return roc_score, pr_score


def interpretability_performance_parallel(X, score, X_model, precomputed_labels=None, n_jobs=-1):
    """
    Parallelized version of interpretability_performance using joblib.

    Args:
        X: Input sequences (N, L, A)
        score: Attribution scores (N, L, A)
        X_model: Ground truth model (N, A, L)
        precomputed_labels: Optional pre-computed labels (N, L)
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        roc_score: Array of AUROC values per sample
        pr_score: Array of AUPR values per sample
    """
    from joblib import Parallel, delayed

    score_abs = np.abs(np.sum(score, axis=2))  # (N, L)

    # Pre-compute labels if not provided
    if precomputed_labels is None:
        gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)
        labels = (gt_info > 0.01).astype(np.float32)
    else:
        labels = precomputed_labels

    def compute_metrics(j):
        gs = score_abs[j]
        label = labels[j]

        precision, recall, _ = precision_recall_curve(label, gs)
        pr = auc(recall, precision)

        fpr, tpr, _ = roc_curve(label, gs)
        roc = auc(fpr, tpr)

        return roc, pr

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_metrics)(j) for j in range(len(score_abs))
    )

    roc_score = np.array([r[0] for r in results])
    pr_score = np.array([r[1] for r in results])

    return roc_score, pr_score
    


def get_callbacks(monitor='val_auroc', patience=20, decay_patience=5, decay_factor=0.2):
    es_callback = keras.callbacks.EarlyStopping(monitor=monitor, 
                                                patience=patience, 
                                                verbose=1, 
                                                mode='max', 
                                                restore_best_weights=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, 
                                                  factor=decay_factor,
                                                  patience=decay_patience, 
                                                  min_lr=1e-7,
                                                  mode='max',
                                                  verbose=1) 

    return [es_callback, reduce_lr]



def compile_model(model):

    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', auroc, aupr])
