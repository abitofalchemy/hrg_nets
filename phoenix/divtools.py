# -*- coding: utf-8 -*-
__author__ = "Rodrigo Palacios, Sal Aguinaga"
__copyright__ = "Copyright 2015, The Phoenix Project"
__credits__ = ["Sal Aguinaga", "Rodrigo Palacios", "Tim Weninger"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Sal Aguinaga"
__email__ = "saguinag (at) nd dot edu"
__status__ = "sa_grph_models"


def avg_jsd2(dists1, dists2):
  ttl_js_div = sum([jsd2(dist1, dists2.get(key, {})) 
                  for key, dist1 in dists1.items()])
  return ttl_js_div/len(dists1)

def jsd2(dist1, dist2, debug=False): #Jensen-shannon divergence
  import warnings
  import numpy as np
  warnings.filterwarnings("ignore", category = RuntimeWarning)

  x = []
  y = []
  if len(dist1) < len(dist2):
        #x = np.append(x, [0 for r in range(dif)])
    for key, val in dist2.items():
      x.append(dist1.get(key, 0))
      y.append(val)
      
  elif len(dist1) >= len(dist2): 
    #y = np.append(y, [0 for r in range(dif)])
    #x = np.array(x)
    for key, val in dist1.items():
      x.append(val)
      y.append(dist2.get(key, 0))
      
  if debug: print('x:', x, 'y:', y)
  x = np.array(x)
  y = np.array(y)
  #print(x,y)
  d1 = x*np.log2(2*x/(x+y))
  d2 = y*np.log2(2*y/(x+y))
  #print(x, y)
  d1[np.isnan(d1)] = 0
  d2[np.isnan(d2)] = 0
  d = 0.5*np.sum(d1+d2)  
  return d
