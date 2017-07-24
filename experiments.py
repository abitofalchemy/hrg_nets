#!/usr/bin/env python
__authors__ = "saguinag"
__email__ = "{saguinag,tweninge,cpennycu} (at) nd dot edu"
__version__ = "0.1.0"

import math
import os
import pickle
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import pandas as pd

#import data.graphs as graphs
#import exact_phrg as xphrg
#import kron_sal as kron
#import net_metrics
#from derive_prules_givenagraph import derive_prules_from

def get_parser():
  parser = ArgumentParser(description='experiments: given an original edgeslist (orig) & a newGList*.p file in <dout>\n'+
                            'e.g., "--newglist mmoreno_vdb_vdb/newGList_100_2.p"')
  parser.add_argument('--orig', metavar='ORIGG', nargs=1, help='Path to the original (edgelist) graph')
  parser.add_argument('--newglist', nargs=1, help='Path to newGList*.p found in the <dout> folder')
  return parser

## Main
"""
parser = get_parser()
args = vars(parser.parse_args())
if not any(args.values()):
    parser.error('No arguments provided.')

orig_fname = args['orig'][0]    # edgelist str
gname = os.path.basename(orig_fname).split('.')
gname = [x for x in gname if not("out" in x)]
gname = [x for x in gname if not("txt" in x)][0]
orig = graphs.load(orig_fname)  # loads a given edgelist
graphName = gname

origG = nx.Graph(orig[0])       # nx graph object
origG.name = graphName          # graph name 'this is key"

tphrg_fname = args['newglist'][0]
tphrgm = pickle.load(open(tphrg_fname, "r"))

DBG = False
print ('# '*40)
print "# Begin xpHRG -< exact phrg graph generation >-"
prules = derive_prules_from([origG])
# Exact PHRG Synthetic Graphs
xphrgm = xphrg.grow_exact_size_hrg_graphs_from_prod_rules(prules[0], origG.name, origG.number_of_nodes(),len(tphrgm))

print "# Begin Kronecker"
if os.path.exists("./tmp_{}-fit2".format(graphName)):
    with open ("./tmp_{}-fit2".format(graphName), 'r') as f:
        lines = f.readlines()
    estim_initiator_mtx = [l.rstrip('\r\n').split('\t')[-1] for l in lines if "Estimated initiator" in l]
    parts= [x for x in estim_initiator_mtx[0].split(';')]
    parts = [x.strip('\[\]') for x in parts]
    top = [float(x) for x in parts[0].split(',')]
    bot = [float(x) for x in parts[1].split(',')]
    # print [top, bot]
    initiator_matrix = [top , bot]
else:
    initiator_matrix = kron.kronfit(origG, graphName)

kronG = []
for x in range(0,len(tphrgm)):
  # kronG.append( product.kronecker_random_graph(11, initiator_matrix) )
  kronG.append( kron.grow_graphs_using_krongen(origG, graphName, initiator_matrix) )
  if DBG: print x,

print
print "# Begin ChungLu"
chungluG = []
for x in range(0, len(tphrgm)):
    chungluG.append( nx.expected_degree_graph(origG.degree().values()) )
    if DBG: print x,

print
print "# Begin BA"
swG = []
print "# ",
for x in range(0, len(tphrgm)):
    swG.append( nx.barabasi_albert_graph (origG.number_of_nodes(), 2) )
    if DBG: print x,

print
print "# Begin Erdos"
erdosG = []
print "# ",
for x in range(0, len(tphrgm)):
    num_e = origG.number_of_edges()
    num_n = origG.number_of_nodes()
    erdosG.append( nx.erdos_renyi_graph (num_n, num_e/ float(num_n*num_n)) )
    if DBG: print x,
"""

print ('# '*40)
def gcd(glist, label):
    print
    print 'GCD', label
    df_g = net_metrics.external_rage(origG,origG.name) # original graph
    # print nx.info(origG)
    results = []
    for i,c in enumerate(glist):
        gcd_network = net_metrics.external_rage(c,'hstar'+str(i))
        # rgfd =  tijana_eval_rgfd(df_g, gcd_network)  ## what is this?
        gcm_g = net_metrics.tijana_eval_compute_gcm(df_g)
        gcm_h = net_metrics.tijana_eval_compute_gcm(gcd_network)
        gcd = net_metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
        if DBG: print '  ', gcd
        results.append(gcd)

    return results


def scale_norm_dist(gb):
    # Scaled
    gb['Sg'] = gb.apply(lambda x: x[0]/float(x.name), axis=1)
    # gb['Sg'] = gb.apply(lambda x: x[1]/float(x[0]), axis=1)
    # Total
    Tg = gb['Sg'].sum()
    # Normalized
    gb['Ng'] = gb['Sg']/float(Tg)
    #orig_sn_deg_dist= gb['Ng'].values
    results = gb['Ng'].values
    return results

def ecdf(x):
  xs = np.sort(x)
  ys = np.arange(1, len(xs)+1)/float(len(xs))
  return xs, ys

def SNDegree_CDF_ks_2samp(glist, origG, label):
    from scipy.stats import ks_2samp
    print
    print 'Scaled-Normalized Degree Dist CDF', label
    kdeg = nx.degree_histogram(origG)[1:]
    df = pd.DataFrame.from_dict(origG.degree().items())
    gb = df.groupby([1]).count()
    ## probability gb['p'] = gb.apply(lambda x: x[0]/float(origG.number_of_nodes()), axis=1)
    # print gb
    # exit()
    origG_sndd = scale_norm_dist(gb)
    # origG_sndd2 = gb.apply(lambda x: x[0]/float(origG.number_of_nodes()), axis=1)
    xs0, ys0 = ecdf(origG_sndd)
    results = []
    for g in glist:
        df = pd.DataFrame.from_dict(g.degree().items())
        gb = df.groupby([1]).count()
        # gb['p'] = gb.apply(lambda x: x[0]/float(g.number_of_nodes()), axis=1)
        # print gb.head()
        # exit()
        hstar_sndd = scale_norm_dist(gb)
        x1, ys2 = ecdf(hstar_sndd)
        # if we ploat x0,ys0 and x1,ys2 ECDF we can see where they
        # diverge most will be shown in D (or the ks_2sample) statistic
        # print origG_sndd[:4]
        # print hstar_sndd[:4]
        # hstar_sndd = gb.apply(lambda x: x[0]/float(g.number_of_nodes()), axis=1)
        # print origG_sndd2[:4]
        # print hstar_sndd[:4]
        # exit()
        D,p = ks_2samp(origG_sndd, hstar_sndd)
        # print "  {}, {}".format(D, p)
        results.append([D,p])

    return results

degree_KS_tst_tphrgm= SNDegree_CDF_ks_2samp	(tphrgm, origG,	'tpHRG')
degree_KS_tst_xphrgm= SNDegree_CDF_ks_2samp	(xphrgm, origG,	'xpHRG')
degree_KS_tst_kronG=  SNDegree_CDF_ks_2samp	(kronG,	origG,	'Kron')
degree_KS_tst_chungluG= SNDegree_CDF_ks_2samp(chungluG,origG, 'chungluG')
degree_KS_tst_swG=    SNDegree_CDF_ks_2samp	(swG, origG, 'BA')
degree_KS_tst_erdosG= SNDegree_CDF_ks_2samp	(erdosG, origG,	'Erdos')

df4 = pd.DataFrame([degree_KS_tst_tphrgm,
                    degree_KS_tst_xphrgm,
                    degree_KS_tst_kronG,
                    degree_KS_tst_chungluG,
                    degree_KS_tst_swG,
                    degree_KS_tst_erdosG])
df4 = df4.transpose()
df4.columns =[  'deg_ks_D_P_tphrg','deg_ks_D_P_xphrg',
                'deg_ks_D_P_Kron','deg_ks_D_P_chungluG',
                'deg_ks_D_P_BA','deg_ks_D_P_Erdos']


def degree_CDF_test(glist, label):
    print
    print 'Degree CDF Test', label
    dist = nx.degree_histogram(origG)[1:]
    s = float(np.sum(dist))
    cnt = float(len(dist))
    cdf = 0
    orig_p = []
    for x in dist:
        wgt = x/s
        cdf += wgt
        orig_p.append(cdf)

    results = []
    for c in glist:
        dist = nx.degree_histogram(c)[1:]
        s = float(np.sum(dist))
        cdf = 0
        g_cdf = []
        for x in dist:
            wgt = x / s
            cdf += wgt
            g_cdf.append(cdf)

        dist = 0
        for i in range(0, min(len(g_cdf), len(orig_p))):
            dist += math.pow(g_cdf[i] - orig_p[i], 2)

        if DBG: print '  ', math.sqrt(dist)
        results.append(math.sqrt(dist))

    return  results
# degree_CDF_test(tphrgm, 'tpHRG')
# degree_CDF_test(kronG, 'Kron')
# degree_CDF_test(chungluG, 'chungluG')
# degree_CDF_test(swG, 'BA')
# degree_CDF_test(erdosG, 'Erdos')







def clustering_CDF_test(glist, label):
    print
    print 'Clustering CDF Test', label
    dist = sorted(nx.clustering(origG).values())
    s = float(np.sum(dist))
    cnt = float(len(dist))
    cdf = 0
    orig_p = []
    for x in dist:
        wgt = x/s
        cdf += wgt
        orig_p.append(cdf)

    results =[]
    for c in glist:
        c = nx.Graph(c)
        dist = dist = sorted(nx.clustering(c).values())
        s = float(np.sum(dist))
        cdf = 0
        g_cdf = []
        for x in dist:
            wgt = x / s
            cdf += wgt
            g_cdf.append(cdf)

        dist = 0
        for i in range(0, min(len(g_cdf), len(orig_p))):
            dist += math.pow(g_cdf[i] - orig_p[i], 2)

        if DBG: print '  ', math.sqrt(dist)
        results.append(math.sqrt(dist))

    return results
# clustering_CDF_test(tphrgm, 'tpHRG')
# clustering_CDF_test(kronG, 'Kron')
# clustering_CDF_test(chungluG, 'chungluG')
# clustering_CDF_test(swG, 'BA')
# clustering_CDF_test(erdosG, 'Erdos')






def eig_CDF_test(glist, label):
    print
    print 'Eig CDF Test', label
    dist = sorted(nx.eigenvector_centrality(origG).values())
    s = float(np.sum(dist))
    cnt = float(len(dist))
    cdf = 0
    orig_p = []
    for x in dist:
        wgt = x/s
        cdf += wgt
        orig_p.append(cdf)

    results =[]
    for c in glist:
        dist = sorted(nx.eigenvector_centrality(c).values())
        s = float(np.sum(dist))
        cdf = 0
        g_cdf = []
        for x in dist:
            wgt = x / s
            cdf += wgt
            g_cdf.append(cdf)

        dist = 0
        for i in range(0, min(len(g_cdf), len(orig_p))):
            dist += math.pow(g_cdf[i] - orig_p[i], 2)

        if DBG: print '  ', type(math.sqrt(dist))
        results.append(math.sqrt(dist))

    return results







def hops_CDF_test(glist, label):
    print
    print 'Hops CDF Test', label
    dist = net_metrics.get_graph_hops(origG, 100).values()
    s = float(np.sum(dist))
    cnt = float(len(dist))
    cdf = 0
    orig_p = []
    for x in dist:
        wgt = x/s
        cdf += wgt
        orig_p.append(cdf)

    results =[]
    for c in glist:
        dist = net_metrics.get_graph_hops(origG, 100).values()
        s = float(np.sum(dist))
        cdf = 0
        g_cdf = []
        for x in dist:
            wgt = x / s
            cdf += wgt
            g_cdf.append(cdf)

        dist = 0
        for i in range(0, min(len(g_cdf), len(orig_p))):
            dist += math.pow(g_cdf[i] - orig_p[i], 2)

        if DBG: print '  ', math.sqrt(dist)
        results.append(math.sqrt(dist))

    return results




# hops_CDF_test(tphrgm, 'tpHRG')
# hops_CDF_test(kronG, 'Kron')
# hops_CDF_test(chungluG, 'chungluG')
# hops_CDF_test(swG, 'BA')
# hops_CDF_test(erdosG, 'Erdos')









def diameter(glist, label):
    print
    print 'Diameter', label
    results = []
    for c in glist:
        bfs_eff_diam= net_metrics.bfs_eff_diam(c, 100, .90)
        if DBG: print bfs_eff_diam
        results.append(bfs_eff_diam)

    return results
# diameter([origG], 'Original')
# diameter(tphrgm, 'tpHRG')
# diameter(kronG, 'Kron')
# diameter(chungluG, 'chungluG')
# diameter(swG, 'BA')
# diameter(erdosG, 'Erdos')

try:
    eig_CDF_tst_tphrgm=eig_CDF_test	(tphrgm,	'tpHRG')
except Exception as e:
    print ( "<p>Error: %s</p>" % e )
    if not ('eig_CDF_tst_tphrgm' in locals()):
      eig_CDF_tst_tphrgm =[]
    pass
try:
    eig_CDF_tst_xphrgm=eig_CDF_test	(xphrgm,	'xpHRG')
except Exception as e:
    print ( "<p>Error: %s</p>" % e )
    if not ('eig_CDF_tst_tphrgm' in locals()):
      eig_CDF_tst_xphrgm =[]
    pass
try:
    eig_CDF_tst_kronG=eig_CDF_test	(kronG,	'Kron')
except Exception as e:
    print ( "<p>Error: %s</p>" % e )
    if not ('eig_CDF_tst_tphrgm' in locals()):
      eig_CDF_tst_kronG =[]
    pass
try:
    eig_CDF_tst_chungluG=eig_CDF_test	(chungluG,	'chungluG')
except Exception as e:
    print ( "<p>Error: %s</p>" % e )
    if not ('eig_CDF_tst_tphrgm' in locals()):
      eig_CDF_tst_chungluG =[]
    pass
try:
    eig_CDF_tst_erdosG=eig_CDF_test	(erdosG,	'Erdos')
except Exception as e:
    print ( "<p>Error: %s</p>" % e )
    if not ('eig_CDF_tst_tphrgm' in locals()):
      eig_CDF_tst_erdosG=[]
    pass
try:
  eig_CDF_tst_swG = eig_CDF_test(swG, 'BA')
except Exception as e:
  print ("<p>Error: %s</p>" % e)
  if not ('eig_CDF_tst_tphrgm' in locals()):
    eig_CDF_tst_swG = []
  pass


df2 = pd.DataFrame([eig_CDF_tst_tphrgm,
                    eig_CDF_tst_xphrgm,
                    eig_CDF_tst_kronG,
                    eig_CDF_tst_chungluG,
                    eig_CDF_tst_swG,
                    eig_CDF_tst_erdosG])
df2 = df2.transpose()
if len(df2.columns) == 6:
    df2.columns = [ 'eig_CDF_tst_tphrgm',
                'eig_CDF_tst_xphrgm',
                'eig_CDF_tst_kronG',
                'eig_CDF_tst_chungluG',
                'eig_CDF_tst_BA',
                'eig_CDF_tst_erdosG' ]
elif len(df2.columns) is 5:
    df2.columns = [
                'eig_CDF_tst_xphrgm',
                'eig_CDF_tst_kronG',
                'eig_CDF_tst_chungluG',
                'eig_CDF_tst_BA',
                'eig_CDF_tst_erdosG' ]


gcd_tphrgm   = gcd(tphrgm, 'tpHRG')
gcd_xphrgm   = gcd(xphrgm, 'xpHRG')
gcd_kronG    = gcd(kronG, 'Kron')
gcd_chungluG = gcd(chungluG, 'chungluG')
gcd_swG      = gcd(swG, 'BA')
gcd_erdosG   = gcd(erdosG, 'Erdos') #Erdos doesn't work with GCD

degree_CDF_tst_tphrgm= degree_CDF_test	(tphrgm, 'tpHRG')
degree_CDF_tst_xphrgm= degree_CDF_test	(xphrgm, 'xpHRG')
degree_CDF_tst_kronG=  degree_CDF_test	(kronG, 'Kron')
degree_CDF_tst_chungluG= degree_CDF_test (chungluG,	'chungluG')
degree_CDF_tst_swG=    degree_CDF_test	(swG, 'BA')
degree_CDF_tst_erdosG= degree_CDF_test	(erdosG, 'Erdos')


clustering_CDF_tst_tphrgm= clustering_CDF_test	(tphrgm,	'tpHRG')
clustering_CDF_tst_xphrgm= clustering_CDF_test	(xphrgm,	'xpHRG')
clustering_CDF_tst_kronG= clustering_CDF_test	(kronG,	'Kron')
clustering_CDF_tst_chungluG= clustering_CDF_test (chungluG,	'chungluG')
clustering_CDF_tst_swG= clustering_CDF_test	(swG,	'BA')
clustering_CDF_tst_erdosG= clustering_CDF_test	(erdosG,	'Erdos')


hops_CDF_tst_tphrgm= hops_CDF_test	(tphrgm,	'tpHRG')
hops_CDF_tst_xphrgm= hops_CDF_test	(xphrgm,	'xpHRG')
hops_CDF_tst_kronG= hops_CDF_test	(kronG,	'Kron')
hops_CDF_tst_chungluG= hops_CDF_test	(chungluG,	'chungluG')
hops_CDF_tst_swG= hops_CDF_test	(swG,	'BA')
hops_CDF_tst_erdosG= hops_CDF_test	(erdosG,	'Erdos')

diameter_tst_origG= diameter	([origG],	'Original')
diameter_tst_xphrgm= diameter	(xphrgm,	'xpHRG')
diameter_tst_tphrgm= diameter	(tphrgm,	'tpHRG')
diameter_tst_kronG= diameter	(kronG,	'Kron')
diameter_tst_chungluG= diameter	(chungluG,	'chungluG')
diameter_tst_swG= diameter(swG,	'BA')
diameter_tst_erdosG= diameter(erdosG,	'Erdos')

#
# ######################################################

df0 = pd.DataFrame(np.transpose([gcd_tphrgm,
                                gcd_xphrgm,
                                gcd_kronG,
                                gcd_chungluG,
                                gcd_swG,
                                gcd_erdosG,
                                degree_CDF_tst_tphrgm,
                                degree_CDF_tst_xphrgm,
                                degree_CDF_tst_kronG,
                                degree_CDF_tst_chungluG,
                                degree_CDF_tst_swG,
                                degree_CDF_tst_erdosG]))

df0.columns = ['gcd_tphrgm',
              'gcd_xphrgm',
              'gcd_kronG',
              'gcd_chungluG',
              'gcd_swG',
              'gcd_erdosG',
              'degree_CDF_tst_tphrgm',
              'degree_CDF_tst_xphrgm',
              'degree_CDF_tst_kronG',
              'degree_CDF_tst_chungluG',
              'degree_CDF_tst_swG',
              'degree_CDF_tst_erdosG']





df3 = pd.DataFrame(np.transpose([
                                clustering_CDF_tst_tphrgm,
                                clustering_CDF_tst_xphrgm,
                                clustering_CDF_tst_kronG,
                                clustering_CDF_tst_chungluG,
                                clustering_CDF_tst_swG,
                                clustering_CDF_tst_erdosG,

                                hops_CDF_tst_tphrgm,
                                hops_CDF_tst_xphrgm,
                                hops_CDF_tst_kronG,
                                hops_CDF_tst_chungluG,
                                hops_CDF_tst_swG,
                                hops_CDF_tst_erdosG,
                                # diameter_tst_origG,
                                diameter_tst_tphrgm,
                                diameter_tst_xphrgm,
                                diameter_tst_kronG,
                                diameter_tst_chungluG,
                                diameter_tst_swG,
                                diameter_tst_erdosG]))


df3.columns = ['clustering_CDF_tst_tphrgm',
              'clustering_CDF_tst_xphrgm',
              'clustering_CDF_tst_kronG',
              'clustering_CDF_tst_chungluG',
              'clustering_CDF_tst_swG',
              'clustering_CDF_tst_erdosG',

              'hops_CDF_tst_tphrgm',
              'hops_CDF_tst_xphrgm',
              'hops_CDF_tst_kronG',
              'hops_CDF_tst_chungluG',
              'hops_CDF_tst_swG',
              'hops_CDF_tst_erdosG',

              'diameter_tst_tphrgm',
              'diameter_tst_xphrgm',
              'diameter_tst_kronG',
              'diameter_tst_chungluG',
              'diameter_tst_swG',
              'diameter_tst_erdosG']


result = pd.concat([df0,  df4, df2, df3], axis=1)
d = np.zeros(result.shape[0])
d.fill(np.nan)
np.put(d, [0],diameter_tst_origG)
result['diameter_eff_origG'] = d#,[np.nan for in range(2,len(result.shape()[0])) ]]


# filelist = [ f for f in os.listdir("./Results") if f.startswith("UNDIR_RESULTS") ]
# for f in filelist:
#     os.remove(f)


result.to_csv("Results/{}.csv".format(graphName))
