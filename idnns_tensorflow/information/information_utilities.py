import numpy as np
num = 1
def KL(a, b):
    """Calculate the Kullback Leibler divergence between a and b """
    D_KL = np.nansum(np.multiply(a, np.log(np.divide(a, b+np.spacing(1)))), axis=1)
    return D_KL

def calc_information_single(PYgivenX, PXs, PYs):
    """"Calculate the MI - I(X;Y)"""
    Hyx = - np.nansum(np.dot(PYgivenX*np.log2(PYgivenX+np.spacing(1)), PXs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IXY = Hy - Hyx
    return IXY

def calc_information_0(probTgivenXs, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT

def calc_information_1(probTgivenXs, PYgivenTs, PXs, PYs, PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    #PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs+np.spacing(1))))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs+np.spacing(1))), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT

def calc_information(probTgivenXs, PYgivenTs, PXs, PYs, PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    #PTs = np.nansum(probTgivenXs*PXs, axis=1)
    t_indeces = np.nonzero(PTs)
    probTgivenXs_copy = np.copy(probTgivenXs)
    PYgivenTs_copy = np.copy(PYgivenTs)
    PYs_copy = np.copy(PYs)
    PTs_copy = np.copy(PTs)

    probTgivenXs_copy[probTgivenXs_copy==0] = np.spacing(1)
    PYgivenTs_copy[PYgivenTs_copy==0] = np.spacing(1)
    PYs_copy[PYs_copy==0] = np.spacing(1)
    PTs_copy[PTs_copy==0] = np.spacing(1)

    # Ht = np.nansum(-np.dot(PTs, np.log2(PTs+np.spacing(1))))
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs_copy)))
    # Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs_copy)), PXs)))
    # Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs_copy), PTs))
    # Hy = np.nansum(-np.dot(PYs, np.log2(PYs+np.spacing(1))))
    Hy = np.nansum(-np.dot(PYs, np.log2(PYs_copy)))

    IYT = Hy - Hyt
    ITX = Ht - Htx

    return ITX, IYT


def t_calc_information(p_x_given_t, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    Hx = np.nansum(-np.dot(PXs, np.log2(PXs)))
    Hxt = - np.nansum((np.dot(np.multiply(p_x_given_t, np.log2(p_x_given_t)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Hx - Hxt
    return ITX, IYT