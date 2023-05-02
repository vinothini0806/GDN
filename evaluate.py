from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def f1_score(y_true, y_pred):
        """
        Calculate the F1 score given the true labels and predicted labels.
        
        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.
        
        Returns:
            f1_score (float): The F1 score.
        """
        tp = 0
        fp = 0
        fn = 0
        # Calculate the number of true positives, false positives, and false negatives.
        tp = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 1) & (y_pred == 0))
        fn = sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            precision = 0.0
            print("precision",precision)
        else:
            precision = tp / (tp + fp)
            print("precision",precision)

        # Calculate precision and recall.
        # precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("recall",recall)
        print("tp",tp)
        print("fn",fn)
        # Calculate the F1 score.
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()
    # calculate error for each column with ground truth and predicted labels
    for i in range(feature_num):
        
        test_re_list = np_test_result[:2,:,i]
        # print("len(test_re_list[0])",len(test_re_list[0]))
        val_re_list = np_val_result[:2,:,i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    
    # print("test_delta",test_delta)
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    # print("len(smoothed_err_scores)",len(smoothed_err_scores))
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    print("len(smoothed_err_scores)",len(smoothed_err_scores))
    return smoothed_err_scores



def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    # print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]
    num_missing_targets = 0
    num_false_targets = 0
    missing_alarm_rate = 0.0
    False_alarm_rate = 0.0
    tp = 0
    tn = 0
    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    # topk_indices cotains top clumns 8 columns
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    # print("len(total_topk_err_scores[0])",len(total_topk_err_scores[0]))
    # final_topk_fmeas is the threshold values and for each thresholds in the thresolds
    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    for i,label in enumerate(gt_labels):
        if label != pred_labels[i]:
            if label==1:
                num_missing_targets += 1
            elif label==0:
                num_false_targets += 1
        else:
            if label==1:
                tn += 1
            elif label==0:
                tp += 1

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    missing_alarm_rate = num_missing_targets/len(gt_labels)
    False_alarm_rate = num_false_targets/len(gt_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold,missing_alarm_rate,False_alarm_rate

