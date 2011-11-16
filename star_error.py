
from __future__ import division
import numpy as np

# cheated and grabbed 95% confidence intervals from http://statpages.org/confint.html
ppois_conf = { 0 : [ 0, 3.6899 ],
               1 : [ .0253, 5.5716 ],
               2 : [ .2422, 7.2247 ],
               3 : [ .6187, 8.7673 ],
               4 : [ 1.0899, 10.2416 ],
               5 : [ 1.6235, 11.6683 ],
               6 : [ 2.2019, 13.0595 ],
               7 : [ 2.8144, 14.4277 ],
               8 : [ 3.4538, 15.7632 ],
               9 : [ 4.1154, 17.0848 ],
               10 : [ 4.7954, 18.3904]}

pois_thresh = 10

def high_low_rate( n_match, n_all):
    rate = n_match / n_all
    if n_match <= pois_thresh:
        low_conf = ppois_conf[n_match][0]
        high_conf = ppois_conf[n_match][1]
        err_low = rate - (low_conf/n_all)
        err_high = (high_conf/n_all) - rate
    else:
        err_low = np.sqrt(n_match)/n_all
        err_high = err_low
        # I don't think the high err limits are necessary on this data set, but
        if err_high + rate > 1:
            err_high = 1 - rate
    return (err_high, err_low)
        
    




