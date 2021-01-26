import torch


def LRG(x, eps=1e-10, undo=False, med = -7.57):
    if not undo:
        #print('LRG: Before the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.log(x + eps, out=x)
        #print('LRG: After the Log: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.sub(x, med, out=x)
        #print('LRG: After the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
    else:
        torch.add(x, med, out=x)
        torch.exp(x, out=x)

def LRD(x, eps=1e-10, undo=False, med = -6.11):
    if not undo:
        #print('LRD: Before the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.log(x + eps, out=x)
        #print('LRD: After the Log: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.sub(x, med, out=x)
        #print('LRD: After the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
    else:
        torch.add(x, med, out=x)
        torch.exp(x, out=x)


def HRG(x, eps=1e-10, undo=False, med = -15.28):
    if not undo:
        #print('HRG: Before the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.log(x + eps, out=x)
        #print('HRG: After the Log: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.sub(x, med, out=x)
        #print('HRG: After the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
    else:
        torch.add(x, med, out=x)
        torch.exp(x, out=x)

def HRD(x, eps=1e-10, undo=False, med = -14.03):
    if not undo:
        #print('HRD: Before the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.log(x + eps, out=x)
        #print('HRD: After the Log: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
        torch.sub(x, med, out=x)
        #print('HRD: After the norm: max is: ' + str(torch.max(x)) + " | Median is: " + str(torch.median(x)))
    else:
        torch.add(x, med, out=x)
        torch.exp(x, out=x)