__version__ = "0.1.0"
import pickle
from argparse import ArgumentParser
from collections import namedtuple
from glob import glob


Clique = namedtuple('Clique', ['rule', 'history', 'nids'])

def examine_generated_graphs_in(dir_path):
  c_files = glob(dir_path+ "/*cliques.p")
  e_files = glob(dir_path+ "/*edges.p")
  for f in c_files:
    clq_lst = pickle.load(open(f, "rb"))
    # print len(clq_lst)

  for f in e_files:
    edg_lst = pickle.load(open(f, "rb"))
    print (edg_lst)
  return 

def get_parser():
  parser = ArgumentParser(description='examine_edges_cliques: Read generated'\
                                      'graphs by time2.py and output' \
                                      'pgfp files in `Results`')
  parser.add_argument("-d", metavar="DIRPATH", required=True,
                      help="[d]irectory path to output of time2")
  # parser.add_argument("-ds", metavar="DIRSUFFIX", required=False,
  #                     help="input file with two matrices")
  # line below checks if input argument is a valid file
  # parser.add_argument("-g", dest="edglstfname", required=True,
  #                     help="input file with two matrices", metavar="EDGLSTFILE",
  #                     type=lambda x: is_valid_file(parser, x))
  # parser.add_argument("-g", dest="edglstfname", required=True,
  #                     help="Name of the graph/network to process.", metavar="EDGLSTFILE")
  return parser

if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())
  args = parser.parse_args()

  directory_in = args.d

  if (directory_in ) is None:
    print parser.print_help()
    exit()

  examine_generated_graphs_in(directory_in)
