import sys
import numpy as np
import random
import numpy.random as nr
import scipy.linalg as sla
from config import Config
from ioutil import *
from scipy.sparse import coo_matrix

def add_zeros_infront_key(diction):
    """
    Add some 0s in front of the number in the KEY of diction.
    And make sure every number has a fixed length. (In this function the length is 7.)
    For Example:
    123-456 ==> 0000123-0000456
    """
    new_dict = {}
    for k,v in diction.iteritems():
        s = '{:07d}-{:07d}'.format(int(k.split('-')[0]),int(k.split('-')[1]))
        new_dict[s] = v
    return new_dict

def remove_zeros_infront_key(L):
    """
    The inverse procedure of add_zeros_infront_key.
    For Example:
    0000123-0000456 ==> 123-456
    """
    new_list = []
    for i in range(len(L)):
        l = ['{}-{}'.format(int(L[i][0].split('-')[0]),int(L[i][0].split('-')[1])), L[i][1]]
        new_list.append(l)
    return new_list

def sort_dict(diction):
    """
    Sort a dictionary with respect to its KEY.
    For Example:
    1-81
    1-123
    3-23
    """
    new_dict = add_zeros_infront_key(diction)
    sorted_list = sorted(new_dict.iteritems(), key=lambda d:d[0])
    return remove_zeros_infront_key(sorted_list)


def genEvenDenseBlock(A, B, p):
    m = []
    for i in xrange(A):
        a = np.random.binomial(1, p, B)
        m.append(a)
    return np.array(m)

def genHyperbolaDenseBlock(A, B, alpha, tau):
    'this is from hyperbolic paper: i^\alpha * j^\alpha > \tau'
    m = np.empty([A, B], dtype=int)
    for i in xrange(A):
        for j in xrange(B):
            if (i+1)**alpha * (j+1)**alpha > tau:
                m[i,j] = 1
            else:
                m[i,j] = 0
    return m

def genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=None, p=1):
    if tau is None:
        tau = A1**alpha * B1**alpha
    m1 = genEvenDenseBlock(A1, B1, p=p)
    m2 = genHyperbolaDenseBlock(A2, B2, alpha, tau)
    M = sla.block_diag(m1, m2)
    return M

def addnosie(M, A, B, p, black=True, A0=0, B0=0):
    v = 1 if black else 0
    for i in xrange(A-A0):
        a = np.random.binomial(1, p, B-B0)
        for j in a.nonzero()[0]:
            M[A0+i,B0+j]=v
    return M


def injectCliqueCamo(M, m0, n0, p, testIdx):
    (m,n) = M.shape
    M2 = M.copy().tolil()

    colSum = np.squeeze(M2.sum(axis = 0).A)
    colSumPart = colSum[n0:n]
    colSumPartPro = np.int_(colSumPart)
    colIdx = np.arange(n0, n, 1)
    population = np.repeat(colIdx, colSumPartPro, axis = 0)

    for i in range(m0):
        # inject clique
        for j in range(n0):
            if random.random() < p:
                M2[i,j] = 1
        # inject camo
        if testIdx == 1:
            thres = p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        if testIdx == 2:
            thres = 2 * p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        # biased camo           
        if testIdx == 3:
            colRplmt = random.sample(population, int(n0 * p))
            M2[i,colRplmt] = 1

    return M2.tocsc()


def generateProps(rates, times, k, s, t0, tsdiffcands, tsp):

    if len(rates) > 0:
        rs = np.random.choice([4, 4.5], size=s)
        if k in rates:
            for r in rs:
                rates[k].append(r)
        else:
            rates[k] = list(rs)
    if len(times) > 0:
        ts = np.random.choice(tsdiffcands, size=s, p=tsp) + t0
        if k in times:
            for t in ts:
                times[k].append(t)
        else:
            times[k] = list(ts)
    return

def injectFraud2PropGraph(freqfile, ratefile, tsfile, acnt, bcnt, goal, popbd,
                          testIdx = 3, idstartzero=True, re=True, suffix=None,
                         weighted=True, output=True):
    config=Config()
    print('injectFraud2PropGraph')
    if not idstartzero:
        print 'we do not handle id start 1 yet for ts and rate'
        ratefile, tsfile = None, None

    M = loadedge2sm(freqfile, coo_matrix, weighted=weighted,
                     idstartzero=idstartzero)
    'smax: the max # of multiedge'
    smax = M.data.max() #max freqency
    if acnt == 0 and re:
        return M, ([], [])
    M2 = M.tolil()
    (m, n) = M2.shape
    rates, times, tsdiffs, t0 = {}, {}, [], 0
    t0, tsdiffcands,tsp = 0, [], []
    #print('ratefile')
    if ratefile is not None:
        rates = loadDictListData(ratefile, ktype=str, vtype=float)
    #print('tsfile')
    if tsfile is not None:
        times = loadDictListData(tsfile, ktype=str, vtype=int)
        tsmin, tsmax = sys.maxint,0
        tsdiffs = np.array([])
        prodts={i:[] for i in xrange(n)}
        for k,v in times.iteritems():
            k = k.split('-')
            pid = int(k[1])
            prodts[pid] += v
        for pv in prodts.values():
            pv = sorted(pv)
            if len(pv)<=2:
                continue
            minv, maxv = pv[0], pv[-1]
            if tsmin > minv:
                tsmin = minv
            if tsmax < maxv:
                tsmax = maxv
            
            vdiff = np.diff(pv)
            'concatenate with [] will change value to float'
            tsdiffs = np.concatenate((tsdiffs, vdiff[vdiff>0]))
        tsdiffs.sort()
        tsdiffs = tsdiffs.astype(int)
	tsdiffcands = np.unique(tsdiffs)[:20] #another choice is bincount
	tsp = np.arange(20,dtype=float)+1
	tsp = 1.0/tsp
	tsp = tsp/tsp.sum()
    #t0 = np.random.randint(tsmin, tsmax,dtype=int)

    colSum = M2.sum(0).getA1()
    colids = np.arange(n, dtype=int)
    targetcands1 = np.argwhere(colSum < popbd).flatten()
    fraudcands1 = np.arange(m,dtype=int) #users can be hacked
    fraudsters={}
    targets={}
    trueA_all=[]
    trueB_all=[]
    for i in range(config.num_block):
        targetcands=np.setdiff1d(targetcands1,trueB_all,assume_unique=True)
        targets[i] = random.sample(targetcands, bcnt)
        camocands = np.setdiff1d(colids, targets[i], assume_unique=True)
        camoprobs = colSum[camocands]/float(colSum[camocands].sum())
        #population = np.repeat(camocands, colSum[camocands].astype(int), axis=0)
    
        fraudcands=np.setdiff1d(fraudcands1,trueA_all,assume_unique=True)
        fraudsters[i] = random.sample(fraudcands, acnt)
        for j in range(acnt):
            trueA_all.append(fraudsters[i][j])
            print('len(trueA_all) %s' % len(trueA_all))
        for j in range(bcnt):
            trueB_all.append(targets[i][j])
            print('len(trueB_all) %s' % len(trueB_all))
        'rating times for one user to one product, multiedge'
        scands = np.arange(1,4,dtype=int)
        sprobs = 1.0/scands
        sprobs = sprobs/sprobs.sum()
        

        # inject near clique
        for j in targets[i]:
            exeusers = random.sample(fraudsters[i], goal)
            for yi in exeusers:
                s = np.random.choice(scands, size=1, p=sprobs)[0] if weighted else 1     
                if (not weighted) and M2[yi,j] > 0:
                    continue
                M2[yi,j] += s
                k = '{}-{}'.format(yi,j)
                generateProps(rates, times, k, s, t0, tsdiffcands,tsp)
        '''
        # inject camo
        p = goal/float(acnt)
        for ui in fraudsters[i]:

            if testIdx == 1:
                thres = p * bcnt / (n - bcnt)
                for j in targets[i]:
                    s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                    if (not weighted) and M2[ui,j] > 0:
                        continue
                    if random.random() < thres:
                        M2[ui,j] += s
                        k = '{}-{}'.format(ui,j)
                        generateProps(rates, times, k, s, t0, tsdiffcands, tsp)
            if testIdx == 2:
                thres = 2 * p * bcnt / (n - bcnt)
                for j in targets[j]:
                    s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                    if (not weighted) and M2[ui,j] > 0:
                        continue
                    if random.random() < thres:
                        M2[ui,j] += s
                        k = '{}-{}'.format(ui,j)
                        generateProps(rates, times, k, s, t0, tsdiffcands, tsp)
            # biased camo           
            if testIdx == 3:
                colRplmt = np.random.choice(camocands, size=int(bcnt*p),p=camoprobs)
                #M2[ui,colRplmt] = 1
                s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                for j in colRplmt:
                    if (not weighted) and M2[ui,j] > 0:
                        continue
                    M2[ui,j] += s
                    k = '{}-{}'.format(ui,j)
                    generateProps(rates, times, k, s, t0, tsdiffcands, tsp)
        '''
        savesm2edgelist(M2.astype(int), freqfile+'.inject'+suffix, idstartzero=idstartzero)
        saveSimpleListData(fraudsters[i], freqfile+'.trueA'+str(i)+suffix)
        saveSimpleListData(targets[i], freqfile+'.trueB'+str(i)+suffix)


    if suffix is not None:
        suffix = str(suffix)
    else:
        suffix =''
    if ratefile is not None and output is True:
        rates_ = sort_dict(rates)
        saveDistListData_List(rates_, ratefile+'.inject'+suffix)
    if tsfile is not None and output is True:
        times_ = sort_dict(times)
        saveDistListData_List(times_, tsfile+'.inject'+suffix)
    M2 = M2.tocoo()
    if not weighted:
        M2.data[0:] = 1
    if output is True:
        savesm2edgelist(M2.astype(int), freqfile+'.inject'+suffix, idstartzero=idstartzero)
        #saveSimpleListData(fraudsters, freqfile+'.trueA'+suffix)
        #saveSimpleListData(targets, freqfile+'.trueB'+suffix)
    if re:
        #print(fraudsters)
        return M2, (fraudsters, targets)
    else:
        return

