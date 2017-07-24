import shelve
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
import re
import argparse
import traceback
import net_metrics as metrics
from glob import glob

__version__ = "0.1.0"
__author__ = ['Salvador Aguinaga']


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

def infinityMirrorGCD(cg, name):
  gn = name
  g  = cg

  df_g = metrics.external_rage(cg)
  gcm_g = metrics.tijana_eval_compute_gcm(df_g)
  clgm_gcd = []
  kpgm_gcd = []

  lst_files = glob("../Results/synthg_*"+ str(name)+ "*th.shl")
  with open ("../Results/gcd_"+ str(name)+ "_inf_mirr.txt", 'w') as f:
    for j,shlf in enumerate(lst_files):
      shlv = shelve.open(shlf)
      # print j, shlf
      # print "=>", len(shlv['clgm'][0]), len(shlv['kpgm'][0])
      f.write('< {} >---- clgm ----\n'.format(j))
      for i,sg in enumerate(shlv['clgm'][0]):
        df = metrics.external_rage(sg)
        gcm_h = metrics.tijana_eval_compute_gcm(df)
        s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
        clgm_gcd.append(s)

      f.write("(" +str(j) +"," + str(np.mean(clgm_gcd)) + ')\n')

          
      print '.'*40
      f.write('< {} >---- kpgm ----\n'.format(j))
      for i,sg in enumerate(shlv['kpgm'][0]):
            df = metrics.external_rage(sg)
            gcm_h = metrics.tijana_eval_compute_gcm(df)
            s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
            kpgm_gcd.append(s)

      f.write("(" +str(j) +"," + str(np.mean(kpgm_gcd)) + ')\n')  
      shlv.close()

def get_parser():
  parser = argparse.ArgumentParser(description='shelves: Process Infinity Mirror Graphs')
  parser.add_argument('--g', metavar='GRAPH', help='graph edge-list')
  # parser.add_argument('--shl', metavar='SHLGRAPHS', help='Shelves: shl file')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


def main():
  global name

  parser = get_parser()
  args = vars(parser.parse_args())

  if not args['g']:
    parser.print_help()
    os._exit(1)

  try:
    cg   = nx.read_edgelist(args['g'])
    # shlv = shelve.open(args['shl'])
  except Exception, e:
    print str(e)

  name = os.path.basename(args['g']).rstrip('.txt')

  # if 0: infinityMirrorGCD(cg, name)


  if 1:
    lst_files = glob("../Results/synthg_*"+ str(name)+ "*1th.shl")
    for j,shlf in enumerate(lst_files):
      shlv = shelve.open(shlf)
      print j, shlf
      print "=>", len(shlv['clgm'][0]), len(shlv['kpgm'][0])

      print '\tdraw_degree_probability_distribution', '-'*40
      metrics.draw_degree_probability_distribution(orig_g_M=[cg], HRG_M=[], pHRG_M=[], chunglu_M=shlv['clgm'][0], kron_M=shlv['kpgm'][0]) #( chunglu_M, HRG_M, pHRG_M, kron_M)
      print '\tdraw_network_value','-'*40
      metrics.draw_network_value([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      print '\tdraw_hop_plot','-'*40
      metrics.draw_hop_plot([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      print '\tdraw_kcore_decomposition','-'*40
      metrics.draw_kcore_decomposition([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      print '\tdraw_clustering_coefficients','-'*40
      metrics.draw_clustering_coefficients([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      print '\tdraw_assortativity_coefficients','-'*40
      metrics.draw_assortativity_coefficients([cg], shlv['clgm'][0], [], [], shlv['kpgm'][0])
      # metrics.draw_diam_plot([], [chunglu_M, HRG_M, pHRG_M, kron_M] )
  #     metrics.draw_degree_rank_plot(G, chunglu_M)
  #     metrics.draw_network_value(G, chunglu_M)
  if 0:
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
  
  print 'Done.'
  return

if __name__ == "__main__":
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    os._exit(1)
  sys.exit(0)
