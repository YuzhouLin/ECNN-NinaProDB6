import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    # labels shape: (X,)
    y = torch.eye(num_classes)
    return y[labels]


def relu_evidence(y):
    return F.relu(y)


def softmax_evidence(y):
    return F.softmax(y, dim=1)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, max=3))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    # if not device:
    #     device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def loglikelihood_loss(y, alpha, device=None):
    # if not device:
    #    device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, params):
    # if annealing_step = 0, no kl
    y = y.to(params['device'])  # 256*12
    alpha = alpha.to(params['device'])  # 256*12
    # S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = loglikelihood_loss(y, alpha, device=params['device'])
    '''
    belief = (alpha-1.0)/S
    batch_n = belief.size()[0]
    u_dis = torch.zeros(batch_n,1).to(device)
    for index_k in range(num_classes):
        temp0 = torch.zeros(batch_n,1).to(device)
        temp1 = torch.zeros(batch_n,1).to(device)
        for index_j in range(num_classes):
            if index_j!=index_k:
                k = belief[:,index_k].reshape(batch_n ,1).to(device)
                j = belief[:,index_j].reshape(batch_n ,1).to(device)
                temp0 += j*(1.0-torch.abs(k-j)/(k+j+1e-8))
                temp1 += j
        u_dis += k*temp0/(temp1+1e-8)
    '''

    if params['kl'] == 0:
        # return loglikelihood_err + loglikelihood_var
        # return 0.6*(loglikelihood_err + loglikelihood_var) + 0.4*u_dis
        return loglikelihood
    elif params['kl'] == 1:
        annealing_coef = \
            torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(
                    params['epoch_num'] / params['annealing_step'],
                    dtype=torch.float32)
            )
        kl_alpha = (alpha - 1) * (1 - y) + 1
        # target_alpha = torch.sum(alpha * y, dim=1, keepdim=True)
        # p_t = target_alpha/S
        # print(target_alpha.size())
        # torch.sum(alpha[y==1],dim=1,keepdim=True)
        # u = num_classes/S

        # A = loglikelihood_err + loglikelihood_var
        # cond_coef = torch.where(loglikelihood_err>0.5,1.0,-1.0)

        # print(cond_coef)
        # loss = A - annealing_coef*(1.-p_t)**2*u

        # loss = A + (loglikelihood_err-0.5)**2*u

        # cond_coef*(1.-p_t)**2*u

        # return loss
        # total_S = torch.sum(alpha,dim=1,keepdim=True)
        # print(total_S)
        # u = u*annealing_coef
        kl_div = annealing_coef * \
            kl_divergence(kl_alpha, params['class_n'], device=params['device'])
        # a = torch.tensor(0.8, dtype=torch.float32)
        return loglikelihood + kl_div  # + (1-p_t)*kl_div
    elif params['kl'] == 2:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        coef = torch.tensor(params['l'], dtype=torch.float32)
        kl_div = coef * \
            kl_divergence(kl_alpha, params['class_n'], device=params['device'])
        return loglikelihood + kl_div


def edl_mse_loss(output, target, params):
    evidence = eval(params['evi_fun'] + '_evidence(output)')
    alpha = evidence + 1
    y = one_hot_embedding(target, params['class_n'])
    loss = torch.mean(mse_loss(y.float(), alpha, params))
    return loss


def cal_recall(outputs, targets):
    _, true_class_n = np.unique(targets, return_counts=True)
    preds = outputs.argmax(dim=1, keepdim=True).numpy()  # (1547, 1)
    recall = []
    for class_index, class_n in enumerate(true_class_n):
        target_each_class = targets == class_index
        pred_result_each_class = preds[target_each_class] == class_index
        recall.append(np.sum(pred_result_each_class) / class_n)
    return recall


def cal_minAP(n_pos, n_neg):
    # theoretical minimum average precision
    AP_min = 0.0
    for i in range(1, n_pos + 1):
        AP_min += i / (i + n_neg)
    AP_min = AP_min / n_pos
    return AP_min


def cal_mis_pm(labels, scores):
    #  calculate the misclassification performance measures
    #  include AUROC, AP, nAP
    #  labels: labels for postive or not
    #  scores: quantified uncertainty; dict
    n_sample = len(labels)  # the total number of predictions
    n_pos = np.sum(labels)  # the total number of positives
    n_neg = n_sample - n_pos  # the total number of negatives
    skew = n_pos / n_sample
    AUROC = {key: [] for key in scores}
    AP = {key: [] for key in scores}
    nAP = {key: [] for key in scores}

    if skew == 0:  # No postive samples found
        #  PR curve makes no sense, record it but dont use it
        for un_type, un_score in scores.items():
            AP[un_type] = float("nan")  # AP
            nAP[un_type] = float("nan")  # normalised AP
    else:
        minAP = cal_minAP(n_pos, n_neg)
        for un_type, un_score in scores.items():
            AUROC[un_type] = metrics.roc_auc_score(labels, un_score)
            AP[un_type] = metrics.average_precision_score(labels, un_score)
            #  normalised AP
            nAP[un_type] = (AP[un_type] - minAP) / (1 - minAP)
    return AUROC, AP, nAP, skew


def cal_entropy(p):
    entropy = np.sum(-p * np.log(p + 1e-8), axis=1, keepdims=True)
    nor_entropy = entropy/(-np.log(1/p.shape[1]))
    #  p.shape[1]: 12
    return nor_entropy


def cal_un_prob(p):
    ##  un_prob: reverse max prob
    #  un_prob = -np.amax(p, axis=1, keepdims=True)
    un_prob = 1-np.amax(p, axis=1, keepdims=True)
    return un_prob


def cal_vacuity(b):  # b: belief
    vacuity = 1 - np.sum(b, axis=1, keepdims=True)
    return vacuity


def cal_dissonance(b):
    dissonance = np.zeros(b.shape[0])
    for i, x_i in enumerate(b.T):
        x_j = np.delete(b, i, axis=1)
        bal = 1 - \
            np.abs(x_j - x_i[:, np.newaxis]) / \
            (x_j + x_i[:, np.newaxis] + 1e-8)
        dissonance += \
            x_i * np.sum(x_j * np.nan_to_num(bal), axis=1) / \
            (np.sum(x_j, axis=1) + 1e-8)

    return dissonance[:, np.newaxis]


def cal_scores(results, edl_used):
    #  results: (samples * classes)
    scores = {}
    if edl_used == 0:  # results are the pred probabilities
        scores['entropy'] = cal_entropy(results)
        #scores['-max_prob'] = -cal_max_prob(results)
        scores['un_prob'] = cal_un_prob(results)
        overall = np.concatenate((scores['entropy'], scores['un_prob']), axis=1)
    else:  # results are the evidences
        alphas = results + 1
        total_evidences = np.sum(alphas, axis=1, keepdims=True)
        probs = alphas / total_evidences
        beliefs = results / total_evidences
        scores['entropy'] = cal_entropy(probs)
        scores['un_prob'] = cal_un_prob(probs)
        scores['vacuity'] = cal_vacuity(beliefs)
        scores['dissonance'] = cal_dissonance(beliefs)
        overall = np.concatenate((scores['entropy'], scores['un_prob'], scores['vacuity'], scores['dissonance']), axis=1)
    
    scores['overall'] = np.max(overall, axis=1, keepdims=True)
    return scores
