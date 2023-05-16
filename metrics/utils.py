import numpy as np

def rowspace_dist(est,target,metric='both'):
    """Compute the angle between two matrices.
    :param np.ndarray est: first matrix (estimated)
    :param np.ndarray target: second matrix (ground truth)
    :param str metric: 'min' or 'avg', avg compute the average distance and min care about the worst case. default : 'min'
    :return: angle between A and B
    """
    _, _, vh_est = np.linalg.svd(est, full_matrices=False)
    _, _, vh_target = np.linalg.svd(target, full_matrices=False)
    dist_matrix = vh_target @ vh_est.T
    dist_matrix = [dist_matrix.squeeze()]
    # print("dist_matrix",dist_matrix)
    if metric == 'avg':
        # return np.linalg.norm(dist_matrix, 'fro')**2/len(vh_target)
        return np.linalg.norm(dist_matrix)**2/len(vh_target)
    elif metric == 'min':
        return (np.linalg.norm(dist_matrix, axis=1).min())**2
    elif metric == 'both':
        return ((np.linalg.norm(dist_matrix, 'fro')**2/len(vh_target), 
                (np.linalg.norm(dist_matrix, axis=1).min())**2))
    else:
        raise ValueError('metric should be either min or avg')
    

def rowspace_dist2(est, target, tol_ratio = 3):
    _, _, vh_est = np.linalg.svd(est, full_matrices=False)

    est_matrix = target @ vh_est.T @ vh_est @ target.T
    upper =  np.linalg.eigvalsh(est_matrix - tol_ratio*target @ target.T).max()
    lower =  np.linalg.eigvalsh(est_matrix - 1/tol_ratio*target @ target.T).min()
    return upper, lower

def compute_relevantSource_similarity(model, target_vector):
    embed_matrx = model.get_full_task_embed_matrix()
    embed_restrict_matrx = model.get_restricted_task_embed_matrix()
    # TODO : might want to add r cond here when target is not single
    v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector, rcond=None)[0]
    v_norm = np.linalg.norm(v)
    v = v/v_norm
    return v