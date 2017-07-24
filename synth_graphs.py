__author__ = 'tweninge'

import subprocess, shlex
from threading import Timer
import math

import networkx as nx
import numpy as np

import treedecomps.net_metrics as metrics
# import HRG
# import treedecomps.PHRG as PHRG
# import product
import pprint as pp
from treedecomps.salPHRG import grow_graphs_using_krongen

def run(cmd, timeout_sec):
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()

def run(cmd, timeout_sec):
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()

# def kronfit(G):
#   """
#
#   Notes:
#   23May16: changes to handle kronfit
#   """
#
#   from os import environ
#
#   with open('tmp.txt', 'w') as tmp:
#       for e in G.edges():
#           tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#
#   if environ['HOME'] == '/home/saguinag':
#     args = ("/home/saguinag/Software/Snap-3.0/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
#   elif environ['HOME'] == '/Users/saguinag':
#     args = ("/Users/saguinag/Theory/MacOSTools/Snap-3.0/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
#   else:
#     args = ('./kronfit.exe -i:tmp.txt -n0:2 -m:"0.9 0.6; 0.6 0.1" -gi:5')
#   #print (args)
#   #args = args.split()
#   #options = {k: True if v.startswith('-') else v
#   #       for k,v in zip(args, args[1:]+["--"]) if k.startswith('-')}
#
#   #proc = subprocess.Popen(shlex.split(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#   """
#   proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#   kill_proc = lambda p: p.kill()
#   timer = Timer(10, kill_proc, [proc])
#   try:
#       timer.start()
#       output, stderr = proc.communicate()
#   finally:
#       timer.cancel()
#
#   print "out"
#   """
#
#   kronout = ""
#   while not kronout:
#     popen = subprocess.Popen(args, stdout=subprocess.PIPE)
#     popen.wait()
#     output = popen.stdout.read()
#
#     top = []
#     bottom = []
#     kronout = output.split('\n')


def kronfit(G):
  """

  Notes:
  23May16: changes to handle kronfit
  """

  from os import environ

  with open('tmp.txt', 'w') as tmp:
      for e in G.edges():
          tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')

  if environ['HOME'] == '/home/saguinag':
    args = ("/home/saguinag/Software/Snap-3.0/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
  elif environ['HOME'] == '/Users/saguinag':
    args = ("/Users/saguinag/ToolSet/Snap-2.4/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
  else:
    args = ('./kronfit.exe -i:tmp.txt -n0:2 -m:"0.9 0.6; 0.6 0.1" -gi:5')
  #print (args)
  #args = args.split()
  #options = {k: True if v.startswith('-') else v
  #       for k,v in zip(args, args[1:]+["--"]) if k.startswith('-')}

  #proc = subprocess.Popen(shlex.split(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  """
  proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  kill_proc = lambda p: p.kill()
  timer = Timer(10, kill_proc, [proc])
  try:
      timer.start()
      output, stderr = proc.communicate()
  finally:
      timer.cancel()

  print "out"
  """

  kronout = ""
  while not kronout:
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()

    top = []
    bottom = []
    kronout = output.split('\n')

  for i in range(0, len(kronout)):
      if kronout[i].startswith("FITTED PARAMS"):
          top = kronout[i + 1].split()
          top = [float(j) for j in top]
          bottom = kronout[i + 2].split()
          bottom = [float(j) for j in bottom]
          break

  if not len(top):
    print 'top:',top,'bottom',bottom
    top = [0,0]
    bottom = [0,0]

  top[1] = bottom[0] = (top[1] + bottom[0]) / 2  # make symmetric by taking average
  fitted = [top, bottom]
  return fitted


def gcd():
    num_nodes = 1000

    ba_G = nx.barabasi_albert_graph(num_nodes, 3)
    er_G = nx.erdos_renyi_graph(num_nodes, .1)
    ws_G = nx.watts_strogatz_graph(num_nodes, 8, .1)
    nws_G = nx.newman_watts_strogatz_graph(num_nodes, 8, .1)

    graphs = [ba_G, er_G, ws_G, nws_G]

    samples = 50

    for G in graphs:
        chunglu_M = []
        for i in range(0, samples):
            chunglu_M.append(nx.expected_degree_graph(G.degree()))

        HRG_M, degree = HRG.stochastic_hrg(G, samples)
        pHRG_M = PHRG.probabilistic_hrg(G, samples)
        kron_M = []
        rmat_M = []
        for i in range(0, samples):
            P = kronfit(G)
            k = math.log(num_nodes, 2)
            kron_M.append(product.kronecker_random_graph(int(math.floor(k)), P, directed=False))

        df_g = metrics.external_rage(G)
        gcd_chunglu = []
        gcd_phrg = []
        gcd_hrg = []
        gcd_kron = []
        for chunglu_M_s in chunglu_M:
            df_chunglu = metrics.external_rage(chunglu_M_s)
            rgfd = metrics.tijana_eval_rgfd(df_g, df_chunglu)
            gcm_g = metrics.tijana_eval_compute_gcm(df_g)
            gcm_h = metrics.tijana_eval_compute_gcm(df_chunglu)
            gcd_chunglu.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
        for HRG_M_s in HRG_M:
            df_hrg = metrics.external_rage(HRG_M_s)
            rgfd = metrics.tijana_eval_rgfd(df_g, df_hrg)
            gcm_g = metrics.tijana_eval_compute_gcm(df_g)
            gcm_h = metrics.tijana_eval_compute_gcm(df_hrg)
            gcd_hrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
        for pHRG_M_s in pHRG_M:
            df_phrg = metrics.external_rage(pHRG_M_s)
            rgfd = metrics.tijana_eval_rgfd(df_g, df_phrg)
            gcm_g = metrics.tijana_eval_compute_gcm(df_g)
            gcm_h = metrics.tijana_eval_compute_gcm(df_phrg)
            gcd_phrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
        for kron_M_s in kron_M:
            df_kron = metrics.external_rage(kron_M_s)
            rgfd = metrics.tijana_eval_rgfd(df_g, df_kron)
            gcm_g = metrics.tijana_eval_compute_gcm(df_g)
            gcm_h = metrics.tijana_eval_compute_gcm(df_kron)
            gcd_kron.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))

        print gcd_chunglu
        print gcd_hrg
        print gcd_phrg
        print gcd_kron
        print
        print

def synth_plots():
    num_nodes = 100
    samples = 5

    chunglu_M = []
    kron_M = []
    HRG_M = []
    pHRG_M = []
    G_M = []

    for i in range(0,samples):
        ##BA Graph
        G = nx.erdos_renyi_graph(num_nodes, .1)
        G_M.append(G)

        for i in range(0, samples):
            chunglu_M.append(nx.expected_degree_graph(G.degree().values()))

        HRG_M_s, degree = HRG.stochastic_hrg(G, samples)
        HRG_M = HRG_M + HRG_M_s
        pHRG_M_s = PHRG.probabilistic_hrg(G, samples)
        pHRG_M = pHRG_M + pHRG_M_s
        for i in range(0, samples):
            P = kronfit(G)
            k = math.log(num_nodes, 2)
            kron_M.append(product.kronecker_random_graph(int(math.floor(k)), P, directed=False))

    metrics.draw_network_value(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)

# --<
# --< main >--
# --<

# 5000 node subgraphs or
# use the entire graph

# edgelists = ["out.sociopatterns-hypertext",
#             "out.sociopatterns-infectious"]
edgelists = ["as20000102_train_1_el.tsv",
             "as-caida20071105_train_1_el.tsv"]
edgelists = ["out.enron",
            "out.dnc-corecipient"]

fn_train = "data/"+edgelists[0]
fn_test  = "data/"+edgelists[1]
Gtrain = nx.read_edgelist(fn_train, comments="%", data=(('ts',int),))
# Gtrain = nx.read_edgelist(fn_train)
Gtrain.name = [x for x in edgelists[0].split('.') if len(x)>3][0]

# Gtst = nx.read_edgelist(fn_test, comments="%", data=(('weight',int),('ts',int),))
Gtst = nx.read_edgelist(fn_test, comments="%", data=(('ts',int),))
# Gtst = nx.read_edgelist(fn_test)
Gtst.name = [x for x in edgelists[0].split('.') if len(x)>3][0]

chunglu_GM = []
kronprd_GM = []

# --<
# --< chung lu graph >--
# --<
z = Gtrain.degree().values()
for i in range(0,50):
    cl_graph = nx.expected_degree_graph(z)
    chunglu_GM.append(cl_graph)
# --<
# --< Kronecker product graph >--
# --<
for i in range(0,50):
    kp_graph = kronfit(Gtrain)
    kronprd_GM.append(kp_graph)

# --< metrics
metricx = ['gcd']
metrics.network_properties( [Gtst], metricx, chunglu_GM, name="clgm_"+Gtst.name, out_tsv=True)
metrics.network_properties( [Gtst], metricx, kronprd_GM, name="kpgm_"+Gtst.name, out_tsv=True)
