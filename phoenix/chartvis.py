__author__ = 'rodrigo'

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import collections


## References:
## * https://networkx.github.io/documentation/latest/examples/drawing/degree_histogram.html
## * http://www.slideshare.net/JoeJoeJoeontheRocks/social-networkanalysisinpython
## * http://www.cl.cam.ac.uk/~an346/day2slides.pdf
## * http://www.cl.cam.ac.uk/~an346/day3slides.pdf
## * https://www.caida.org/research/topology/generator/

##
## TODO: Degree distribution (y-axis) v Node count rather than node ids
##       since node ids are meaningless at this point.
##  [x] done
## TODO: Assortativity v Degree (in reverse order)


def Assortativity_Plot(orig_g, multGraphs, graph_name, xs):
  """
  Assortativity, or assortative mixing is a preference for a network's nodes to attach to others that are similar in some way.
  (from Wikipedia)
  - Newman, Phys. Rev. E 67, 026126 (2003).
  """
  r_assortative_mixing =[]
  for G in  multGraphs:
    #r_assortative_mixing.append({G.number_of_nodes(): nx.degree_assortativity_coefficient(G)})
    r_assortative_mixing.append( nx.degree_assortativity_coefficient(G))

  df = pd.DataFrame(r_assortative_mixing)
  # print r_assortative_mixing[:4]
  print max(df.index)

  xs.scatter(df.index, df[0], alpha=0.5)
  xs.plot(max(df.index)+1, nx.degree_assortativity_coefficient(orig_g), 'ro')


  return

def hops(all_succs, start, level=0, debug=False):
  """
  T. Weninger
  """

  if debug: print("level:", level)

  succs = all_succs[start] if start in all_succs else []
  if debug: print("succs:", succs)

  lensuccs = len(succs)
  if debug: print("lensuccs:", lensuccs)
  if debug: print()
  if not succs:
    yield level, 0
  else:
    yield level, lensuccs

  for succ in succs:
    #print("succ:", succ)
    for h in hops(all_succs, succ, level+1):
      yield h

def bfs_eff_diam(G, NTestNodes, P):
  EffDiam = -1
  FullDiam = -1
  AvgSPL = -1

  DistToCntH = {}

  NodeIdV = nx.nodes(G)
  random.shuffle(NodeIdV)


  for tries in range(0, min(NTestNodes, nx.number_of_nodes(G)) ):
    NId = NodeIdV[tries]
    b = nx.bfs_successors(G, NId)
    for l, h in hops(b, NId):
      if h is 0: continue
      if not l+1 in DistToCntH:
        DistToCntH[l+1] = h
      else:
        DistToCntH[l+1] += h

  DistNbrsPdfV = {}
  SumPathL=0.0
  PathCnt=0.0
  for i in DistToCntH.keys() :
    DistNbrsPdfV[i] = DistToCntH[i]
    SumPathL += i*DistToCntH[i]
    PathCnt += DistToCntH[i]

  oDistNbrsPdfV = collections.OrderedDict(sorted(DistNbrsPdfV.items()))

  CdfV = oDistNbrsPdfV
  for i in range(1,len(CdfV)):
    if not i+1 in CdfV:
      CdfV[i+1] = 0
    CdfV[i+1] = CdfV[i] + CdfV[i+1]

  EffPairs = P * CdfV[next(reversed(CdfV))]

  for ValN in CdfV.keys():
    if CdfV[ValN] > EffPairs : break

  if ValN >= len(CdfV): return next(reversed(CdfV))
  if ValN is 0: return 1
  #interpolate
  if nx.__version__ == '1.9.1':
    DeltaNbrs = CdfV.items()[ValN] - CdfV.items()[ValN-1];
  else:
    DeltaNbrs = CdfV[ValN] - CdfV[ValN-1];
  if DeltaNbrs is 0: return ValN;
  return ValN-1 + (EffPairs - CdfV[ValN-1])/DeltaNbrs


def hops(all_succs, start, level=0, debug=False):
  if debug: print("level:", level)

  succs = all_succs[start] if start in all_succs else []
  if debug: print("succs:", succs)

  lensuccs = len(succs)
  if debug: print("lensuccs:", lensuccs)
  if debug: print()
  if not succs:
    yield level, 0
  else:
    yield level, lensuccs

  for succ in succs:
    #print("succ:", succ)
    for h in hops(all_succs, succ, level+1):
      yield h

def get_graph_hops(graph):
  from collections import Counter
  from random import sample

  c = Counter()
  node = sample(graph.nodes(), 1)[0]
  b = nx.bfs_successors(graph, node)


  for l, h in hops(b, node):
    c[l] += h

  return c

##
## DRAW FUNCTIONS
##
def draw_degree_rank_plot(orig_g, mG, g1_label, ax):
  import matplotlib.patches as mpatches
  from matplotlib.gridspec import GridSpec
  from numpy import array

  gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

  ori_degree_seq=sorted(nx.degree(orig_g).values(),reverse=True) # degree sequence
  deg_seqs = []
  for newg in mG:
    deg_seqs.append(sorted(nx.degree(newg).values(),reverse=True)) # degree sequence
  df = pd.DataFrame(deg_seqs)

  ax.loglog(df.mean(),'b', label="mean")
  #ax.fill_between(df.columns, df.mean()+2*df.std(), df.mean()-2*df.std(), color='b', alpha=0.2,label="$2\sigma$")
  if nx.__version__ == '1.10':
    ax.fill_between(df.columns, df.mean()-df.sem(), df.mean()+df.sem(), color='r', alpha=0.2, label="se")
  ax.loglog(ori_degree_seq,'ko', mfc='none')

  #### LEGEND
  b_patch  = mpatches.Patch(color='blue',  label="mean")
  lb_patch = mpatches.Patch(color='b', label="2$\sigma$",alpha=0.2)
  lr_pacth  = mpatches.Patch(color='r', label="se",alpha=0.2)
  or_pacth  = mpatches.Patch(color='k', label=g1_label, fill=False)
  ax.legend(handles=[b_patch, lb_patch, lr_pacth, or_pacth])

##  ax.set_yscale('log')
##  ax.set_yscale('log')
  ax.set_title("Degree Distribution v Rank")
  ax_y = ax.set_ylabel('Degree')
  ax_x = ax.set_xlabel('Rank')
#  offset = array([0,.75])
#  ax.yaxis.set_label_coords(0.06, 0.1)
#
#  print ax_x.get_position()
#  ax.xaxis.set_label_coords(0.1, 0.06)
#  ax.xaxis.labelpad = -5
#  ax.yaxis.set_label_coords(0.05, 0.85)


#  blue_patch  = mpatches.Patch(color='blue',  label=g1_label)
#  green_patch = mpatches.Patch(color='black', label="Synthetic")
#  ax.legend(handles=[blue_patch, green_patch])
#  ax.legend()

#

def draw_scree_plot(orig_g, mG,g1_label, ax):
  """
  draw_scree_plot:

  A Scree Plot is a simple line segment plot that shows the fraction of total variance in the data as explained or represented by each PC. The PCs are ordered, and by definition are therefore assigned a number label, by decreasing order of contribution to total variance.
  """
  import numpy.linalg
  import matplotlib.patches as mpatches
  from matplotlib.gridspec import GridSpec
  import math

  # print "draw_scree_plot"
  gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

  L1=nx.normalized_laplacian_matrix(orig_g)
  e1 = np.linalg.eigvals(L1.A)

#  U, S, V = np.linalg.svd(L1.A)
#  eigvals = S**2 / np.cumsum(S)[-1]
#  eigvals2 = S**2 / np.sum(S)
#  assert (eigvals == eigvals2).all()

  #fig = plt.figure(figsize=(8,5))
#  sing_vals = np.arange(len(eigvals)) + 1
#  ax.plot(sing_vals, eigvals, 'ro-', linewidth=2)
#  ax.plot(sing_vals, eigvals2, 'go-', linewidth=2)

  ## Notes: The eigen values from the laplacian spectrum might be far more
  ## accurate
  #e1 = nx.laplacian_spectrum(orig_g)
  #ax.plot(sorted(e1, reverse=True), 'k') # histogram with 100 bins
  #ax.title('Scree Plot')
  #plt.xlabel('Principal Component')
  #plt.ylabel('Eigenvalue')

  gsscr = []
  for g in mG:
    L2=nx.normalized_laplacian_matrix(g)
    e2= np.linalg.eigvals(L2.A)
    gsscr.append([abs(x) for x in sorted(e2, reverse=True)])

  df = pd.DataFrame(gsscr)

  ax.plot(df.columns, abs(df.mean()),'b')
#   ax.fill_between(df.columns, abs(df.mean())-2*abs(df.std()), abs(df.mean())+2*abs(df.std()), color='b', alpha=0.2)
  if nx.__version__ == '1.10':
    ax.fill_between(df.columns, df.mean()-df.sem(), df.mean()+df.sem(), color='r', alpha=0.2)
  ax.plot(sorted(e1, reverse=True), 'ko', mfc='none') # histogram with 100 bins

#  #### LEGEND
#  b_patch  = mpatches.Patch(color='blue',  label="mean")
#  lb_patch = mpatches.Patch(color='b', label="2$\sigma$",alpha=0.2)
#  lr_pacth  = mpatches.Patch(color='r', label="se",alpha=0.2)
#  or_pacth  = mpatches.Patch(color='k', label=g1_label, fill=False)
#  ax.legend(handles=[b_patch, lb_patch, lr_pacth, or_pacth])

  #### AXES LABELS
  ax.set_xlabel('Rank')
  ax.set_title('Scree Plot')
  ax.set_ylabel('Eigenvalues')
#  ax.xaxis.labelpad = -5
#  ax.yaxis.set_label_coords(0.05, 0.85)
  # set plot to log log scale
  ax.set_yscale('log')
  ax.set_xscale('log')
  ax.set_ylim(0.1,10)


def draw_network_value(orig_g, mG, g1_label, ax):
  """
  Network values: The distribution of eigenvector components (indicators of "network value") 
  associated to the largest eigenvalue of the graph adjacency matrix has also been found to be 
  skewed (Chakrabarti et al., 2004).
  """
  import matplotlib.patches as mpatches
  from matplotlib.gridspec import GridSpec
  import pandas as pd
  gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

  eig_cents = [nx.eigenvector_centrality_numpy(g) for g in mG] # nodes with eigencentrality

  srt_eig_cents = sorted(eig_cents, reverse=True)
  net_vals = []
  for cntr in eig_cents:
    net_vals.append(sorted(cntr.values(), reverse=True))
  df = pd.DataFrame(net_vals)
  ax.plot(df.columns, df.mean(),'b')
#   ax.fill_between(df.columns, df.mean()-2*df.# std(), df.mean()+2*df.std(), color='b', alpha=0.2)
  if nx.__version__ == '1.10':
    ax.fill_between(df.columns, df.mean()-df.sem(), df.mean()+df.sem(), color='r', alpha=0.2)
  ax.plot(sorted(nx.eigenvector_centrality(orig_g).values(), reverse=True), 'bo', mfc='none')

  #### AXES LABELS
  ax.set_yscale('log') # set plot to log log scale
  ax.set_xscale('log')

  ax.set_ylabel('Network value')
  ax.set_xlabel('Rank')
  ax.set_title('Network value \n distribution')
#  ax.xaxis.labelpad = -5
#  ax.yaxis.set_label_coords(0.05, 0.85)
#  ax.plot(range(orig_g.number_of_nodes()), es, 'ko',alpha=0.5 )
#    ax.plot(sorted(cntr.values(), reverse=True), 'g', alpha=0.4)
#
#    ax.set_title("Network Value Distribution")

#
#    blue_patch = mpatches.Patch(color='blue', label=g1_label)
#    green_patch = mpatches.Patch(color='green', label="Graph 2")
#    ax.legend(handles=[blue_patch, green_patch])
#
##    plt.show()


def draw_hop_plot(orig_g, mG, g1_label, ax):
  """
    hops object is missing

  """
  import matplotlib.patches as mpatches
  from matplotlib.gridspec import GridSpec

  m_hops_ar = []
  for g in mG:
    c = get_graph_hops(g)
    d = dict(c)
    m_hops_ar.append(d.values())

  df = pd.DataFrame(m_hops_ar)
#  print np.shape(f)
#  print l
##  print f
#  print [f.mean(), f.std(), f.sem()]
#  print f
#  exit()
#  ax.plot(f.columns, f.mean())
#
#  df = pd.DataFrame(m_hops_ar, columns=['x','y'])
#  result = df.pivot(index='x', values='y')
#  print result.head()
#    break
#    cmplx_tups = zip(*list(get_graph_hops(orig_g).items()))
#    x,y = zip(*list(get_graph_hops(orig_g).items()))
    #m_hops_ar.append( zip(*list(get_graph_hops(orig_g).items())))
#    m_hops_ar.append()
#    m_hops_ar.append([*zip(*list(get_graph_hops(orig_g).items())))
#
#  dat = [*x for x in m_hops_ar]

## Notes:
## http://stackoverflow.com/questions/15891038/pandas-change-data-type-of-columns
## Some weird behavior about the way zip deals with the tuples or the way
## DataFrame deals witmy array ... it can't handle a large puple/arr in order
## to convert it to numeric ... so I can do mean or std or sem on it.
## I upgraded pandas ... and still `convert_objects` or `pd.to_numeric` don't
## work. For example ...
##  print pd.Series(m_hops_ar[1]).mean()
##  returns: "TypeError: Could not convert (0, 1, 2, 3, 4, 2, 20, 10, 1, 0) to numeric"

#  print np.shape(m_hops_ar)
#  print m_hops_ar[0][0]
#  print stats.sem(m_hops_ar[0])
#
#  df = pd.DataFrame(m_hops_ar,columns=['x','y'])
#  df['av'] = [np.mean(x) for x in df['y']]
#  df['sd'] = [np.std(x) for x in df['y']]
#
#  print
#  print df.head()
  #df = pd.to_numeric(df[['x','y']])
  #df = df.convert_objects(convert_numeric=True)
#  df[['x','y']] = df[['x','y']].astype(int)
#  df["x"] = df[0].astype("numeric")
#  df["y"] = df[1].astype("numeric")
#  df = df.convert_objects(convert_numeric=True)
#  print df.dtypes
#  print df.head()
#  sr = df['y']
#  print stats.mean(sr.iloc)
#  sr = pd.to_numeric(sr)
##  print df.shape
#  print sr.tail(),sr.mean(), sr.std(), sr.sem
#  exit()
  ax.plot(df.mean(),'b')
#   ax.fill_between(df.columns, abs(df.mean())-2*df.# std(), abs(df.mean())+2*df.std(), color='b', alpha=0.2)
  if nx.__version__ == '1.10':
    ax.fill_between(df.columns, df.mean()-df.sem(), df.mean()+df.sem(), color='r', alpha=0.2)

  ## original plot
  c = get_graph_hops(orig_g)
  d = dict(c)
  ax.plot(d.keys(), d.values(), 'k-o', mfc='none')

#  ax.plot(*zip(*list(get_graph_hops(graph2).items())), c='#009900', alpha=0.4)
#
#  ax.set_title("Hop plot (neighbors v hops)")
  ax.set_ylabel('Rechable Pairs, r(h)')
  ax.set_xlabel('Nbr of Hops, h')
  ax.set_title('Hop Plot')

#  ax.xaxis.labelpad = -5
#  ax.yaxis.set_label_coords(0.05, 0.5)
#  ax.yaxis.labelpad = -3

#  ax.set_yscale('log')
#  ax.set_xscale('log')
#
#  blue_patch = mpatches.Patch(color='blue', label=g1_label)
#  green_patch = mpatches.Patch(color='green', label="Synthetic")
#  plt.legend(handles=[blue_patch, green_patch])

#    plt.show()


def draw_transitivity_nodecounts(graphs, g1_label, ax):
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

#    fig1 = plt.figure(figsize=[16,5])

    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

#    ax1 = fig1.add_subplot(gs[:,0:45])

    nodecount_transitivity = [(len(g.nodes()), nx.transitivity(g)) for g in graphs]

    ax.plot(*zip(*nodecount_transitivity), c='#0000FF')

#    ax.set_title("Transitivity v Node count")
#    ax.set_ylabel('Transitivity')
    ax.set_xlabel('Node Count')


    #### LEGEND
    blue_patch = mpatches.Patch(color='blue', label=g1_label)
    ax.legend(handles=[blue_patch])

#    plt.show()

#def running_mean(x, N):
#  cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
#  return (cumsum[N:] - cumsum[:-N]) / N

def draw_nodecount_degreecount_plots(graphs, g1_label, ax):
  from matplotlib.gridspec import GridSpec
#  import matplotlib.gridspec as gridspec

  gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

  nodecount_degreecount = [(len(g.nodes()), len(g.edges())) for g in graphs]
  df = pd.DataFrame(nodecount_degreecount, columns=['vcnt','ecnt'])

#  df['vcs'] = np.cumsum(df.vcnt, axis=0, out=df.vcnt)
#  df['ecs'] = np.cumsum(df.ecnt, axis=0, out=df.ecnt)
#  df['vmav'] = [x[1]/(x[0]+1) for x in df['vcs'].iteritems()]
#  df['emav'] = [x[1]/(x[0]+1) for x in df['ecs'].iteritems()]

#  ax.plot(df['vcnt'],  df['ecnt'], 'ko', alpha=0.5)
##  ax.plot(df.vmav, df.emav, 'b.', alpha=0.5)
#  df['vcnt'].plot.box(ax=ax, vert=False)
  ## right hand side y-axis
  GRID = False
  if GRID:
    gs = GridSpec(2, 2,
                           width_ratios=[1,2],
                           height_ratios=[4,1]
                           )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax2.plot(df['vcnt'],  df['ecnt'], 'ko', alpha=0.5)
    df['vcnt'].plot.box(ax=ax4, vert=False)
    df['ecnt'].plot.box(ax=ax1)
  else:
    ax.plot(df['vcnt'],  df['ecnt'], 'ko', alpha=0.5)

    # Save the default tick positions, so we can reset them..
    tcksx = ax.get_xticks()
    tcksy = ax.get_yticks()

    ax.boxplot(df['ecnt'], positions=[min(tcksx)], notch=True, widths=1.)
    ax.boxplot(df['vcnt'], positions=[min(tcksy)], vert=False, notch=True, widths=1.)

    ax.set_yticks(tcksy) # pos = tcksy
    ax.set_xticks(tcksx) # pos = tcksx
    ax.set_yticklabels([int(j) for j in tcksy])
    ax.set_xticklabels([int(j) for j in tcksx])
    ax.set_ylim([min(tcksy-1),max(tcksy)])
    ax.set_xlim([min(tcksx-1),max(tcksx)])

#  d2 = pd.DataFrame.from_dict(degree_cnt)
#  ax2 = ax.twinx()
#
#  ax2.plot(d2.columns, d2.mean(),'b')
#  ax2.fill_between(d2.columns, d2.mean()-2*d2.std(), d2.mean()+2*d2.std(), color='b', alpha=0.2)
#  ax2.fill_between(d2.columns, d2.mean()-d2.sem(), d2.mean()+d2.sem(), color='r', alpha=0.2)
#  ax2.set_ylabel('Degree',color='blue')

  #exit()
  #ax.plot(*zip(*nodecount_degreecount))

  ax.set_title("Edge v Node Counts")
  ax.set_ylabel('Edge Count')
  ax.set_xlabel('Node Count')

#  ax.xaxis.labelpad = -5
#  ax.yaxis.labelpad = -40
#  ax.yaxis.set_label_coords(0.075, 0.85)
#  ax2.yaxis.set_label_coords(0.95, 0.85)

#plt.show()

def draw_nodecount_avgdegree_plots(graphs, g1_label, ax):
  from matplotlib.gridspec import GridSpec
  import matplotlib.patches as mpatches

#    fig1 = plt.figure(figsize=[16,5])

  gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

#    ax1 = fig1.add_subplot(gs[:,0:45])

  nodecount_avgdegree = [(len(g.nodes()), sum(nx.degree(g).values())/len(g.nodes())) for g in graphs]

  df = pd.DataFrame(nodecount_avgdegree, columns=['vcnt','eavgcnt'])
  print df.head()
  ax.plot(df['vcnt'],  df['eavgcnt'], '+')

#    ax.plot(*zip(*nodecount_avgdegree))

#  ax.set_title("Avg. Degree v Node Count")
  ax.set_ylabel('Avg. Degree')
  ax.set_xlabel('Node count')
#  ax.yaxis.set_label_coords(0.05, 0.85)

#    plt.show()


def draw_laplacian_spectrum_conf_bands(orig_g, synth_graphs, graph_name, ax):
  """
  ax = axis handle for a given figure

  Reference:
  http://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
  """
  import matplotlib.patches as mpatches
  import pandas as pd

  eigs = [nx.laplacian_spectrum(g) for g in synth_graphs]
  df = pd.DataFrame(eigs)

  es = nx.laplacian_spectrum(orig_g)
  ax.plot(df.min(), 'k:')
  ax.plot(df.max(), 'k:')
  ax.fill_between(df.columns, df.mean()-df.quantile(.05), df.mean()+df.quantile(.95), color='b', alpha=0.2)
  if nx.__version__ == '1.10':
    ax.fill_between(df.columns, df.mean()-df.sem(), df.mean()+df.sem(), color='r', alpha=0.2)
  ax.plot(range(orig_g.number_of_nodes()), es, 'ko',alpha=0.5 )

#  ax.plot(df.quantile(.95),'r--',alpha=0.5)
#  ax.plot(df.quantile(.05),'r--',alpha=0.5)
#  ax.fill_between(df.columns, df.mean()-2*df.std(), df.mean()+2*df.std(), color='b', alpha=0.2)
#
#  ax.set_title("Laplacian Spectrum")
#  blue_patch = mpatches.Patch(color='r', label=g1_label)
#  green_patch = mpatches.Patch(color='black', label="Synthetic")
#  ax.legend(handles=[blue_patch, green_patch])

  #### AXES LABELS
  # set plot to log log scale
#  ax.set_yscale('log')
#  ax.set_xscale('log')

  ## Set
  ax.set_ylabel('Eigenvalues')
  ax.set_xlabel('Node Count')
#  ax.yaxis.set_label_coords(0.05, 0.85)

  return

def draw_laplacian_spectrum(orig_g, new_g, g1_label, ax):
  """
  The set of all N Laplacian eigenvalues is called the Laplacian spectrum of a graph G. The second smallest eigenvalue is lamdaN >= 1, but equal to zero only if a graph is disconnected. Thus, the multiplicity of 0 as an eigenvalue of Q is equal to the number of components of G [8].
  """

  import matplotlib.patches as mpatches
  es = nx.laplacian_spectrum(new_g)
  ax.plot(range(new_g.number_of_nodes()), es, 'k-o', alpha=0.2)
  es = nx.laplacian_spectrum(orig_g)
  ax.plot(range(orig_g.number_of_nodes()), es, 'r-o')


  ax.set_title("Laplacian Spectrum")
  blue_patch = mpatches.Patch(color='r', label=g1_label)
  green_patch = mpatches.Patch(color='black', label="Synthetic")
  ax.legend(handles=[blue_patch, green_patch])

  # set plot to log log scale
  ax.set_yscale('log')
  ax.set_xscale('log')

def draw_clustering_coefficients(orig_g, mG, g1_label, xs):
  import matplotlib.patches as mpatches

  ## clustering coefficients
  ## nx clustering and degree yield values for each node
  method = False

  if method:
    dg_ar = []
    cc_ar = []
    for g in mG:
      df = pd.DataFrame()
      df['cc'] = nx.clustering(g).values()
      df['dg'] = nx.degree(g).values()
      df.sort_values(by=['dg'],ascending=False, inplace=True)
      dg_ar.append(df.dg.values)
      cc_ar.append(df.cc.values)
      
    df0 = pd.DataFrame(dg_ar)
    df1 = pd.DataFrame(cc_ar)

    ax.plot(sorted(df0.mean().values), \
      sorted(df1.mean().values, reverse=True), 'bo', alpha=0.8)
  else:
    df0 = pd.DataFrame()
    for g in mG:
      #df['clco'] = pd.DataFrame.from_dict(nx.clustering(g).items())
      #df['degr'] = pd.DataFrame.from_dict(nx.degree(g).items())
      df = pd.DataFrame()
      df['cc'] = nx.clustering(g).values()
      df['dg'] = nx.degree(g).values()
      df0 = df0.append(df)

  if nx.__version__ == '1.10':
    df0.sort_values(by='cc',ascending=False, inplace=True)
  
  ## Group by
  gb = df0.groupby(['dg'])
  df = gb.mean()
  xs.plot([int(a) for a in df.index], df.cc.values, 'b')
  if nx.__version__ == '1.10':
    sedf = gb.sem()
#     sddf = gb.std()
    ## Plot Intervals
    xs.fill_between([int(a) for a in df.index], df.cc.values-sedf.cc.values, \
                  df.cc.values+sedf.cc.values, color='r', alpha=0.2) ## plot degree in reverse order (with unordered v.ids)
  # xs.fill_between([int(a) for a in sddf.index], df.cc.values-2*sddf.cc.values, \
  #                 df.cc.values+2*sddf.cc.values, color='b', alpha=0.2)
  ## for the reference graph:
  # cc= sorted(nx.clustering(orig_g).values(), reverse=True)
  # xs.plot(cc, 'o',markerfacecolor='gray', markeredgecolor='k', alpha=0.75)

  ## AXES | TITLE | LEGEND
  #  blue_patch = mpatches.Patch(color='r', label=g1_label)
  #  green_patch = mpatches.Patch(color='black', label="Synthetic")
  #  plt.legend(handles=[blue_patch, green_patch])
  xs.set_title("Clustering Coefficients")
  xs.set_ylabel('cc (avg)')
  xs.set_xlabel('degree k')
  #xs.yaxis.set_label_coords(0.05, 0.85)
  xs.yaxis.labelpad = -5



def draw_degree_distribution(origGraph, multiGraphs, glabel, ax):
  """
  draw_degree_distribution:

  origGraph  = original or reference graph
  multGraphs = synthetic graphs
  glabel     = graph label or name
  ax         = figure/plot axis

  http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/

  """
  import matplotlib.patches as mpatches

  deg_dst = []
  mdf = pd.DataFrame()
  for g in multiGraphs:
    run_nbr = multiGraphs.index(g)
    dg_dict = nx.degree(g)
    deg_dst.append(sorted(dg_dict, reverse=True))

  mdf = pd.DataFrame(deg_dst)

  ax.plot(mdf.mean(), 'b') ## plt deg in revrs order (w unordered v.ids)
#   ax.fill_between(mdf.columns, mdf.mean()-2*mdf.std(), mdf.mean()+2*mdf.std(), color='b', alpha=0.2) ## plot degree in reverse order (with unordered v.ids)
  if nx.__version__ == '1.10':
    ax.fill_between(mdf.columns, mdf.mean()-mdf.sem(),mdf.mean()+mdf.sem(), color='r', alpha=0.2) ## plot degree in reverse order (with unordered v.ids)

  ## special handling of the x-axis
  vids = False
  if vids:
    axs = ax.get_xticks()
    x = [np.where(xtick_labels==i)[0]  for i in axs]
    x = [item for sublist in x for item in sublist]
    ax.set_xticklabels(x)
    #ndf.plot(y=['degdis'], color='b',kind='scatter')


  ##
  ## NB: Degree (in reverse order) v node ids (not in order)
  ##

  ax.set_xlabel('Nodes')
  ax.set_ylabel('Degree')
  ax.set_title('Degree Distribution')
  #  ax.yaxis.set_label_coords(0.05, 0.85)
  ax.set_xscale('log')
  ax.set_yscale('log')

#  print ym

#  gb = mdf.groupby(['v'])
#  print gb.groups

#  degree_dis = [nx.degree(g) for g in multGraphs]
#  degree_dis = nx.degree(origGraph)
#  df = pd.DataFrame.from_dict(degree_dis.items())
#  df.columns = ['v','deg']
#  print df.head()
#  df = df.groupby(['deg']).count()
##  print df.count()

  ## list of dictionaries
  ## where ea dict is the degree distribution per run
#  degree_dicts = [nx.degree(g) for g in multiGraphs]
#
#  df = pd.DataFrame([dd.items() for dd in degree_dicts])
#  # df is not colums of tuples (node, degree) where column = run
#  print 'df\n', df.head()
#  x = range((df.shape)[1])
#  print  df.iteritems()
#  ndf = pd.DataFrame(tmp)
#  ym = ndf.mean()
#  ys = ndf.std()
#  ye = ndf.sem()
#
#  print np.shape(x), np.shape(ym)
#  ax.plot(x, ym,'b')
#  ax.fill_between(x, ym-ys, ym+ys, color='b', alpha=0.2)

#  print df.shape
#  for i in np.arange(0,np.shape(df)[1]):
#    degs.append(df[i][1].mean(axis=1))
#
#  print degs[:4]
#  exit()
#
#  for dv in degree_dist:
#    tdf = pd.DataFrame.from_dict(dv.items())
#    df = df.append([tdf])

#  df.columns = ['v','deg']
#  df.sort('deg', ascending=False, inplace=True )
##  print "tdf.groupby(['v'])"
#  gbd = df.groupby(['v'])
#  print gbd.groups.values().head()
#
#  ndf = pd.DataFrame(gbd.groups.items())
#  df.columns = ['v','deg']
#  ndf = ndf.sort(columns=['deg'],ascending=False, inplace=True)
#  print df.head(),'\n', df.shape
#  exit()
#
#  yp = gbd.mean()+gbd.std()
#  ym = gbd.mean()-gbd.std()
#
#  #print len(gbd.mean()-gbd.std()), len(gbd.groups.keys()), np.shape(x), np.shape(ym)
#  #print ym.head()
#  ax.fill_between(gbd.groups.keys(), ym['deg'], yp['deg'], color='b', alpha=0.2)
#  yp = gbd.mean()+gbd.sem()
#  ym = gbd.mean()-gbd.sem()
#  ax.fill_between(gbd.groups.keys(), ym['deg'], yp['deg'], color='r', alpha=0.2)
#  ax.plot(gbd.mean(),'b')



#  ## plot the mean, std, and SE.
#  ax.fill_between(ddf.index, ddf.mean(axis=1)-ddf.std(axis=1), ddf.mean(axis=1)+ ddf.std(axis=1), color='b', alpha=0.2)
#  ax.fill_between(ddf.index, ddf.mean(axis=1)-ddf.sem(axis=1), ddf.mean(axis=1)+ ddf.sem(axis=1), color='r', alpha=0.2)
#  ax.loglog(ddf.index, ddf.mean(axis=1),'b')
