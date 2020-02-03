import numpy as np 

R_left =np.array([[2,2,1],
                  [2,1,0],
                  [1,1,1]])

R_right =np.array([[1,1,1],
                   [0,1,2],
                   [1,2,2]])

def trangular_fuzzifier(slope, delta):
    if delta < 0:
        y1 = min(-delta/slope,1)
        y2 = max(delta/slope + 1,0)
        y3 = 0
    elif delta ==0:
        y1 = 0
        y2 = 1
        y3 = 0
    elif delta>0:
        y1 = 0
        y2 = max(-delta/slope + 1,0)
        y3 = min(delta/slope,1)

    return (y1, y2, y3)

def fuzzification(delta_e, delta_d):
    e = trangular_fuzzifier(0.1,delta_e)
    d = trangular_fuzzifier(0.1,delta_d)
    v_l = np.zeros(3)
    v_r = np.zeros(3)

    for i in range(3):
        for j in range(3):
            imp = min(e[i],d[j])
            if v_l[R_left[i][j]]<imp:
                v_l[R_left[i][j]] = imp
            if v_r[R_right[i][j]]<imp:
                v_r[R_right[i][j]] = imp

    return v_l, v_r

def defuzzification(v):
    """
        minmum of maxmum defuzzifier 
    """
    index = np.where(v == np.max(v))[0][0]
    if index == 0:
        return (-1*v[0])
    elif index == 1:
        return 0.2*(v[1] - 1)
    elif index == 2:
        return (1*v[2])
    