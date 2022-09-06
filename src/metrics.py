import torch
import numpy as np

# pred : prediction
# gt : ground truth

def get_tf_pn(pred, gt, ts=[]):
    ts = [t.lower() for t in ts]
    res = []
    for w in ts:
        # true-positive
        if w=='tp':
            r = ((pred==1)+(gt==1))==2
        # true-negetive
        elif w=='tn':
            r = ((pred==0)+(gt==0))==2
        # false-positive
        elif w=='fp':
            r = ((pred==1)+(gt==0))==2
        # false-negetive
        elif w=='fn':
            r = ((pred==0)+(gt==1))==2
        res.append(r)
    return res
         

def get_accuracy(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    corr = torch.sum(pred==gt)
    tensor_size = pred.size(0)*pred.size(1)*pred.size(2)*pred.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(pred, gt, threshold=0.5):
    # Sensitivity == Recall
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tp, fn = get_tf_pn(pred, gt, ['tp', 'fn'])
    se = float(torch.sum(tp))/(float(torch.sum(tp+fn)) + 1e-6)     
    return se


def get_specificity(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tn, fp = get_tf_pn(pred, gt, ['tn', 'fp'])
    sp = float(torch.sum(tn))/(float(torch.sum(tn+fp)) + 1e-6)
    return sp


def get_precision(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tp, fp = get_tf_pn(pred, gt, ['tp', 'fp'])
    pc = float(torch.sum(tp))/(float(torch.sum(tp+fp)) + 1e-6)
    return pc


def get_f1(pred, gt, threshold=0.5):
    # Sensitivity == Recall
    se = get_sensitivity(pred, gt, threshold=threshold)
    pc = get_precision(pred, gt, threshold=threshold)
    f1 = 2*se * pc/(se+pc + 1e-6)
    return f1


def get_js(pred, gt, threshold=0.5):
    # js : jaccard similarity
    pred = pred > threshold
    gt = gt == torch.max(gt)
    Inter = torch.sum((pred+gt)==2)
    Union = torch.sum((pred+gt)>=1)
    js = float(Inter)/(float(Union) + 1e-6)
    return js


def get_dc(pred,gt,threshold=0.5):
    # dc : dice coefficient
    pred = pred > threshold
    gt = gt == torch.max(gt)
    Inter = torch.sum((pred+gt)==2)
    dc = float(2*Inter)/(float(torch.sum(pred)+torch.sum(gt)) + 1e-6)
    return dc



class Metrics(object):

    def __init__(self, pred, gt, threshold=0.5):
        self.pred = pred > threshold
        self.gt   = gt == torch.max(gt)
        self.tp, self.tn, self.fp, self.fn = self.get_tf_pn()


    def get_tf_pn(self):
        # true-positive
        tp = ((self.pred==1)+(self.gt==1))==2
        # true-negetive
        tn = ((self.pred==0)+(self.gt==0))==2
        # false-positive
        fp = ((self.pred==1)+(self.gt==0))==2
        # false-negetive
        fn = ((self.pred==0)+(self.gt==1))==2
        return tp, tn, fp, fn


    def get_accuracy(self):
        corr = torch.sum(self.pred==self.gt)
        tensor_size = torch.tensor(np.prod(list(self.pred.shape)))
        acc = float(corr)/float(tensor_size)
        return acc


    def get_sensitivity(self):
        se = float(torch.sum(self.tp))/(float(torch.sum(self.tp+self.fn)) + 1e-6)     
        return se


    def get_specificity(self):
        sp = float(torch.sum(self.tn))/(float(torch.sum(self.tn+self.fp)) + 1e-6)
        return sp


    def get_precision(self):
        pc = float(torch.sum(self.tp))/(float(torch.sum(self.tp+self.fp)) + 1e-6)
        return pc


    def get_f1(self):
        se = self.get_sensitivity()
        pc = self.get_precision()
        f1 = 2*se * pc/(se+pc + 1e-6)
        return f1


    def get_js(self):
        # js : jaccard similarity
        Inter = torch.sum((self.pred+self.gt)==2)
        Union = torch.sum((self.pred+self.gt)>=1)
        js = float(Inter)/(float(Union) + 1e-6)
        return js


    def get_dc(self):
        # dc : dice coefficient
        Inter = torch.sum((self.pred+self.gt)==2)
        spg = torch.sum(self.pred)+torch.sum(self.gt)
        dc = float(2*Inter)/(float(spg) + 1e-6)
        return dc