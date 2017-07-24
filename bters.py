import shelve
import networkx as nx
import pandas as pd
import numpy as np
import math
import os
import sys
import re
import argparse
import traceback
import net_metrics as metrics
from glob import glob


__version__ = "0.1.0"
__author__ = ['Salvador Aguinaga']

# alchemee analyze the BTER generated graphs

def get_basic_stats(grphs,gen_mod, name):

  df = pd.DataFrame()
  for g in grphs:
    tdf = [pd.Series(g.degree().values()).mean(), pd.Series(nx.clustering(g).values()).mean()]
    df = df.append([tdf])
  df.columns=['avg_k','avg_cc']
  df.to_csv()


def get_degree_dist(grphs,gen_mod, name):
  mf = pd.DataFrame()
  for g in grphs:
    d = g.degree()
    df = pd.DataFrame.from_dict(d.items())
    gb = df.groupby([1]).count()
    mf = pd.concat([mf, gb], axis=1)

  mf['pk'] = mf.mean(axis=1)/float(g.number_of_nodes())
  mf['k'] = mf.index.values
  #print mf
  out_tsv = '../Results/{}_{}_degree.tsv'.format(name,gen_mod)
  mf[['k','pk']].to_csv(out_tsv, sep='\t', index=False, header=True, mode="w")

def get_clust_coeff(grphs,gen_mod, name):
  mf = pd.DataFrame()
  for g in grphs:
    df = pd.DataFrame.from_dict(g.degree().items())
    df.columns=['v','k']
    cf = pd.DataFrame.from_dict(nx.clustering(g).items())
    cf.columns=['v','cc']
    df = pd.merge(df,cf,on='v')
    mf = pd.concat([mf, df])
  gb = mf.groupby(['k']).mean()


  out_tsv = "../Results/{}_{}_clustering.tsv".format(name,gen_mod)
  gb[['cc']].to_csv(out_tsv, sep="\t", header=True, index=True)

def degree_prob_distributuion(orig_g_M, otherModel_M, name):
    print 'draw degree probability distribution'

    if orig_g_M is not None:
      dorig = pd.DataFrame()
      for g in orig_g_M:
        d = g.degree()
        df = pd.DataFrame.from_dict(d.items())
        gb = df.groupby(by=[1])
        dorig = pd.concat([dorig, gb.count()], axis=1)  # Appends to bottom new DFs
      print "---<>--- orig", name
      if not dorig.empty :
        zz = len(dorig.mean(axis=1).values)
        sa =  int(math.ceil(zz/75))
        if sa == 0: sa=1
        for x in range(0, len(dorig.mean(axis=1).values), sa):
            print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

    if otherModel_M is not None:
      dorig = pd.DataFrame()
      for g in otherModel_M:
        d = g.degree()
        df = pd.DataFrame.from_dict(d.items())
        gb = df.groupby(by=[1])
        dorig = pd.concat([dorig, gb.count()], axis=1)  # Appends to bottom new DFs
      print "---<>--- otherModel_M", name
      if not dorig.empty :
        zz = len(dorig.mean(axis=1).values)
        sa =  int(math.ceil(zz/float(75)))
        for x in range(0, len(dorig.mean(axis=1).values), sa):
            print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"


def network_value_distribution(orig_g_M, otherModel_M, name):
  eig_cents = [nx.eigenvector_centrality_numpy(g) for g in orig_g_M]  # nodes with eigencentrality
  net_vals = []
  for cntr in eig_cents:
      net_vals.append(sorted(cntr.values(), reverse=True))
  df = pd.DataFrame(net_vals)

  print "orig"
  l = list(df.mean())
  zz = float(len(l))
  if not zz == 0:
      sa =  int(math.ceil(zz/75))
      for i in range(0, len(l), sa):
          print "(" + str(i) + "," + str(l[i]) + ")"

  eig_cents = [nx.eigenvector_centrality_numpy(g) for g in otherModel_M]  # nodes with eigencentrality
  net_vals = []
  for cntr in eig_cents:
      net_vals.append(sorted(cntr.values(), reverse=True))
  df = pd.DataFrame(net_vals)

  print "other model"
  l = list(df.mean())
  zz = float(len(l))
  if not zz == 0:
      sa =  int(math.ceil(zz/75))
      for i in range(0, len(l), sa):
          print "(" + str(i) + "," + str(l[i]) + ")"

def hop_plots(orig_g_M, otherModel_M, name):
  
  m_hops_ar = []
  for g in orig_g_M:
      c = metrics.get_graph_hops(g, 20)
      d = dict(c)
      m_hops_ar.append(d.values())
  df = pd.DataFrame(m_hops_ar)
  print '-- orig graph --\n'

  l = list(df.mean())
  zz = float(len(l))
  if not zz == 0:
      sa =  int(math.ceil(zz/float(75)))
      for i in range(0, len(l), sa):
          print "(" + str(i) + "," + str(l[i]) + ")"


  print '-- the other model --\n'
  m_hops_ar = []
  for g in otherModel_M:
      c = metrics.get_graph_hops(g, 20)
      d = dict(c)
      m_hops_ar.append(d.values())
      break

  df = pd.DataFrame(m_hops_ar)
  l = list(df.mean())
  zz = float(len(l))
  if not zz == 0:
      sa =  int(math.ceil(zz/float(75)))
      for i in range(0, len(l), sa):
          print "(" + str(i) + "," + str(l[i]) + ")"

def clustering_coefficients(orig_g_M, otherModel_M, name):
  if len(orig_g_M) is not 0:
    dorig = pd.DataFrame()
    for g in orig_g_M:
      degdf = pd.DataFrame.from_dict(g.degree().items())
      ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
      dat = np.array([degdf[0], degdf[1], ccldf[1]])
      df = pd.DataFrame(np.transpose(dat))
      df = df.astype(float)
      df.columns = ['v', 'k', 'cc']

      dorig = pd.concat([dorig, df])  # Appends to bottom new DFs

    print "orig"
    gb = dorig.groupby(['k'])
    zz = len(gb['cc'].mean().values)
    sa =  int(math.ceil(zz/75))
    if sa == 0: sa=1
    for x in range(0, len(gb['cc'].mean().values), sa):
        print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"



  if len(otherModel_M) is not 0:
    dorig = pd.DataFrame()
    for g in otherModel_M:
        degdf = pd.DataFrame.from_dict(g.degree().items())
        ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
        dat = np.array([degdf[0], degdf[1], ccldf[1]])
        df = pd.DataFrame(np.transpose(dat))
        df = df.astype(float)
        df.columns = ['v', 'k', 'cc']

        dorig = pd.concat([dorig, df])  # Appends to bottom new DFs

    print "otherModel_M"
    gb = dorig.groupby(['k'])
    zz = len(gb['cc'].mean().values)
    sa =  int(math.ceil(zz/75))
    if sa == 0: sa=1
    for x in range(0, len(gb['cc'].mean().values), sa):
        print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"
  return

def assortativity(orig_g_M, otherModel_M, name):
  if len(orig_g_M) is not 0:
    dorig = pd.DataFrame()
    for g in orig_g_M:
        kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
        kcdf['k'] = g.degree().values()
        dorig = pd.concat([dorig, kcdf])

    print "orig"
    gb = dorig.groupby(['k'])
    zz = len(gb[1].mean().values)
    sa =  int(math.ceil(zz/75))
    if sa == 0: sa=1
    for x in range(0, len(gb[1].mean().values), sa):
        print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

  if len(otherModel_M) is not 0:
      dorig = pd.DataFrame()
      for g in otherModel_M:
          kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
          kcdf['k'] = g.degree().values()
          dorig = pd.concat([dorig, kcdf])

      print "the other model ", name
      gb = dorig.groupby(['k'])
      zz = len(gb[1].mean().values)
      sa =  int(math.ceil(zz/75))
      if sa == 0: sa=1
      for x in range(0, len(gb[1].mean().values), sa):
          print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

  return

def kcore_decomposition(orig_g_M, otherModel_M, name):
  dorig = pd.DataFrame()
  for g in orig_g_M:
      g.remove_edges_from(g.selfloop_edges())
      d = nx.core_number(g)
      df = pd.DataFrame.from_dict(d.items())
      df[[0]] = df[[0]].astype(int)
      gb = df.groupby(by=[1])
      dorig = pd.concat([dorig, gb.count()], axis=1)  # Appends to bottom new DFs
  print "orig"

  if not dorig.empty :
      zz = len(dorig.mean(axis=1).values)
      sa =  int(math.ceil(zz/75))
      if sa == 0: sa=1
      for x in range(0, len(dorig.mean(axis=1).values), sa):
          print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

  dorig = pd.DataFrame()
  for g in otherModel_M:
      d = nx.core_number(g)
      df = pd.DataFrame.from_dict(d.items())
      df[[0]] = df[[0]].astype(int)
      gb = df.groupby(by=[1])
      dorig = pd.concat([dorig, gb.count()], axis=1)  # Appends to bottom new DFs
  print "== the other model =="
  if not dorig.empty :
      zz = len(dorig.mean(axis=1).values)
      sa =  int(math.ceil(zz/75))
      if sa == 0: sa=1
      for x in range(0, len(dorig.mean(axis=1).values), sa):
          print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
  return

def alchemee(graph,graphName):
  g  = graph
  gn = graphName
  lst_files = glob("../BTERgraphs/*{}*th.tsv".format(gn))
  for j,f in enumerate(lst_files):
    print '--<{}>-- {} --'.format(j,f)

    a  = nx.read_edgelist(f) 
    # degree_prob_distributuion( [g], [a], gn)
    # print '-- network value --'
    # network_value_distribution([g], [a], gn)
    # print '-- Hop Plot --'
    # hop_plots([g], [a], gn)
    # print '\tclustering coeffs -- \n'
    # clustering_coefficients([g], [a], gn)
    print '\tdraw_assortativity_coefficients -- \n'
    assortativity([g], [a], gn)
    # print '\tdraw_kcore_decomposition -- \n'
    # kcore_decomposition([g], [a], gn)
  return


def get_parser():
  parser = argparse.ArgumentParser(description='shelves: Process Infinity Mirror Graphs')
  parser.add_argument('--g', metavar='GRAPH', help='graph edge-list')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


def main():
  global name

  parser = get_parser()
  args = vars(parser.parse_args())

  if not args['g']:
    parser.print_help()
    os._exit(1)
  
  print args['g']
  try:
    cg   = nx.read_edgelist(args['g'])
    # shlv = shelve.open(args['shl'])
  except Exception, e:
    print str(e)
    cg   = nx.read_edgelist(args['g'], comments="%")

  name = os.path.basename(args['g']).rstrip('.txt')
  
  if 1:
    alchemee(cg, name)
    print 'alchemee: Done'
    exit(0)


  if 1:
    lst_files = glob("../Results/synthg_*"+ str(name)+ "*.shl")

    for j,shlf in enumerate(lst_files):
      shlv = shelve.open(shlf)
      print "====>", j, len(shlv['clgm'][0]), len(shlv['kpgm'][0]), len(shlv['kpgm'][0][0]), type(shlv['kpgm'][0][0])

      # print '\tdraw_degree_probability_distribution', '-'*40
      # metrics.draw_degree_probability_distribution(orig_g_M=[cg], HRG_M=[], pHRG_M=[], chunglu_M=shlv['clgm'][0], kron_M=shlv['kpgm'][0]) #( chunglu_M, HRG_M, pHRG_M, kron_M)
      # print '\tdraw_network_value','-'*40
      # metrics.draw_network_value([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])

      # print '\tdraw_hop_plot','-'*40
      # metrics.draw_hop_plot([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      # print '\tdraw_kcore_decomposition','-'*40
      # metrics.draw_kcore_decomposition([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      # print '\tdraw_clustering_coefficients','-'*40
      # metrics.draw_clustering_coefficients([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      # print '\tdraw_assortativity_coefficients','-'*40
      # metrics.draw_assortativity_coefficients([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      # metrics.draw_diam_plot([], [chunglu_M, HRG_M, pHRG_M, kron_M] )
  #     metrics.draw_degree_rank_plot(G, chunglu_M)
  #     metrics.draw_network_value(G, chunglu_M)
  #     print '-- degree dist --'
  #     degree_prob_distributuion( [cg], shlv['kpgm'][0], name)
  #     print '-- network value --'
      # network_value_distribution([cg], shlv['kpgm'][0], name)
      # print '-- Hop Plot --'
      # hop_plots([cg], [shlv['kpgm'][0][0]], name)
      # print '-- clustering coeffs --'
      # clustering_coefficients([cg], shlv['kpgm'][0], name)
      # print '\tdraw_assortativity_coefficients','-'*40
      # assortativity([cg], shlv['kpgm'][0], name)
      print '\tdraw_kcore_decomposition','-'*40
      kcore_decomposition([cg], shlv['kpgm'][0], name)

  else:
    lst_files = glob("../Results/*"+ str(name)+ "*.shl")
    with open('../Results/{}_gcd_infinity.txt'.format(str(name)), 'w') as tmp:
      tmp.write('-- {} ----\n'.format(name))
      for j,shlf in enumerate(lst_files):
        print "--"+ shlf + "-"*40
        shlv = shelve.open(shlf)
        df_g = metrics.external_rage(cg)
        gcm_g = metrics.tijana_eval_compute_gcm(df_g)
        clgm_gcd = []
        kpgm_gcd = []

        tmp.write('---- clgm ----\n')
        for i,sg in enumerate(shlv['clgm'][0]):
          df = metrics.external_rage(sg)
          gcm_h = metrics.tijana_eval_compute_gcm(df)
          s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
          #
          # tmp.write("(" + str(i) + "," + str(s) + ')\n')
          clgm_gcd.append(s)

        tmp.write("(" +str(j) +"," + str(np.mean(clgm_gcd)) + ')\n')

        tmp.write('---- kpgm ----\n')
        for i,sg in enumerate(shlv['kpgm'][0]):
          df = metrics.external_rage(sg)
          gcm_h = metrics.tijana_eval_compute_gcm(df)
          s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
          #
          # tmp.write("(" + str(i) + "," + str(s) + ')\n')
          kpgm_gcd.append(s)

        tmp.write("(" +str(j) +"," + str(np.mean(kpgm_gcd)) + ')\n')


if __name__ == "__main__":
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    os._exit(1)
  sys.exit(0)
