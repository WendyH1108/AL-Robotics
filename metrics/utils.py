import numpy as np
from dataset.utils import generate_pendulum_specified_kernel

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
    # print("dist_matrix",dist_matrix)
    if metric == 'avg':
        return np.linalg.norm(dist_matrix, 'fro')**2/len(vh_target)
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

def most_related_source(model, target_vector, true_v, task_dim, domain='ball', task_aug_kernel=None, already_compute = None):

    est_input_embed_matrix = model.get_input_embed_matrix()
    est_input_embed_matrix, s, vh = np.linalg.svd(est_input_embed_matrix, full_matrices=False)
    embed_matrix = model.get_full_task_embed_matrix()
    embed_restrict_matrix = model.get_restricted_task_embed_matrix()
    embed_matrix = np.diag(s) @ vh @ embed_matrix
    embed_restrict_matrix = np.diag(s) @ vh @ embed_restrict_matrix

    if domain == 'pendulum':
        if already_compute is None:
            assert task_aug_kernel is not None, "task_aug_kernel should be provided for pendulum domain"
            sample_num = 10**task_dim

            diff = 1
            iter = 0
            target_vector = task_aug_kernel(target_vector.T)
            v = 0
            while diff > 0.01 and iter < 6:
                np.random.seed()
                tmp = np.random.uniform(-1,1, (sample_num, task_dim ))*0.5**iter + v
                tmp[:,-1] = 0
                tmp_aug = task_aug_kernel(tmp)
                best_ind = np.linalg.norm(embed_matrix @ tmp_aug.T - embed_restrict_matrix @ target_vector.T, axis=0).argmin()
                v = tmp[[best_ind]]
                diff = np.linalg.norm(embed_matrix @ tmp_aug[[best_ind]].T - embed_restrict_matrix @ target_vector.T, axis=0)
                print(diff)
                iter += 1
        else:
            v = np.expand_dims(already_compute, axis=0)
        est_v = task_aug_kernel(v).T # (d_W_aug, 1)
        est_v = est_v/np.linalg.norm(est_v)
        similarity = est_v.T @ true_v
    else:
        # TODO : might want to add r cond here when target is not single
        # print("est B_W v_target", embed_matrix @ target_vector)
        v = np.linalg.lstsq(embed_restrict_matrix, embed_matrix @ target_vector, rcond=None)[0]
        v_norm = np.linalg.norm(v)
        est_v = v/v_norm
        similarity = est_v.T @ true_v

    return similarity, est_v

def compute_matrix_spectrum(w, aug_kernel):
    """
    :param w: (n, d_W)
    """
    tmp = aug_kernel(w)[:,:-1]
    eigs = np.linalg.eigvals(tmp.T @ tmp)
    return np.max(eigs), np.min(eigs), np.max(eigs)/np.min(eigs)