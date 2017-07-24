__authors__ = 'saguinag,tweninge,dchiang'
__contact__ = '{authors}@nd.edu'
__version__ = "0.1.0"

# infinitymirror.py  

# VersionLog:
# 0.1.0 Initial state; 

import glob
import matplotlib
import networkx as nx
import pandas   as pd
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import net_metrics as metrics



def draw_inf_degree_probability_distribution(orig_g,
                                             mG=None,
                                             axs=None,
                                             io_color=None,
                                             m_color=None,
                                             gname=''):
  with open('../Results/inf_deg_dist_{}.txt'.format(gname), 'w') as f:

    d = orig_g.degree()
    n = orig_g.number_of_nodes()

    df = pd.DataFrame.from_dict(d.items())
    df.columns = ['v', 'k']
    gb = df.groupby(by=['k'])

    if axs is None:
      f, axs = plt.subplots(1, 1, figsize=(1.6 * 6, 1 * 6))
    x = gb.count().index.values
    y = gb.count().values / float(n)

    if 'phrg' in gname:
      axs.plot(x, y, '-o', color='k')  # plot distribution of original graph
      f.write('# original graph\n')
      for i in range(len(y)):
        # print i,x[i],y[i] #'({}, {})\n'.format(x[i],y[i])
        f.write('({}, {})\n'.format(x[i], y[i]))
      # else:
      #   axs.plot(x,y, ':o', color=o_color)    # plot distribution of original graph


    if mG is not None:
      col_names = []
      multigraph_df = pd.DataFrame()
      for i, hstar in enumerate(mG):
        d = hstar.degree()
        n = len(d)
        df = pd.DataFrame.from_dict(d.items())
        gb = df.groupby(by=[1])
        col_names.append('H*_{}'.format(i))
        multigraph_df = pd.concat([multigraph_df, gb.count()], axis=1)  # Appends to bottom new DFs

      cdf = multigraph_df / float(n)

    # print type (cdf)
    cdf.columns = col_names

    # cdf.plot(ax=axs, colormap='Greens_r', marker='o', linestyle=':', alpha=0.8)

    axs.plot(cdf.index, cdf.mean(axis=1), ':.', color=m_color, label=gname)
    # one sigma
    axs.fill_between(cdf.index, cdf.mean(axis=1) -cdf.std(axis=1), cdf.mean(axis=1) +cdf.std(axis=1), color=m_color, alpha=0.2)

    #   ## special case
    f.write('# average deg dist from 10 recursions {}\n'.format(gname))
    yy = cdf.mean(axis=1).values
    for i,mu in enumerate(yy):
      f.write('({}, {})\n'.format(cdf.index[i], mu))


    axs.set_ylabel(r"$p(k)$")
    axs.set_xlabel(r"degree, $k$")
    # axs.patch.set_facecolor('None')
    # # special case:
    # axs.set_ylim(orig_floor*0.9,1.0)
    axs.set_xlim(0.9,10**2)

    axs.grid(True, which='both')
    axs.spines['left'].set_color('#B0C4DE')
    axs.yaxis.tick_left()
    axs.spines['bottom'].set_color('#B0C4DE')
    axs.xaxis.tick_bottom()
    axs.set_yscale('log')
    axs.set_xscale('log')


# ~~~~~~~~~~~~~~~~
# * Main - Begin *

# inf_files_prefix = ['as_phrg', 'as_kpgm','as_clgm']
G = nx.karate_club_graph()
#G = nx.read_edgelist("../demo_graphs/as20000102.txt")
inf_files_prefix = ['as_phrg', 'as_kpgm','as_chlu']
inf_files_prefix = ['synth_as_phrg', 'synth_as_kpgm','synth_as_chlu']

f, axs = plt.subplots(1, 1, figsize=(1.6 * 6., 1 * 6.))

for prefix in inf_files_prefix:

  gpickle_files = glob.glob('../Results/'+prefix +'*.gpickle')
  print '../Results/'+prefix +'*.gpickle'
  mGraphs = []
  for i, f in enumerate(gpickle_files):
    g = nx.read_gpickle(f)
    if str(9) in f:
      mGraphs.append(g)



  # metrics.draw_degree_probability_distribution(G, mGraphs, axs=axs, gname='as_kpgm')
  draw_inf_degree_probability_distribution(G, mGraphs, axs=axs, gname=prefix)

plt.legend(labels=['Orig Graph', r'PHRG, $\^{\mu}$', 'Kron, $\^{\mu}$','Chung-Lu , $\^{\mu}$'])
plt.title(r'KC - Degree Dist. for after the $10^{th}$ recursion')
# axs.patch.set_facecolor('lightgray')
plt.savefig('outfig', bb_inches='tight')
plt.close()
print 'Done'
