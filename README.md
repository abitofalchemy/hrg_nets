# hrg_nets
Exploring the limits of HRG for network modeling
# PhoenixPython

## StarLog (Version Log)

16Nov16 |`toy.py` reading cindy rules fails in g.sample 
         Walk throug the parsing it might do the thing

09Jan16 |New sampler (in david.py) that should use much much less memory.


## From Tree Decomp to Prod Rules
To print the the Tree:
	in `tree_decomposition`, line 202 (`new_visit`) 
To print the Prod Rules:
	in `tree_decomposition`, line 299 (`add_to_prod_rules`)

## Main Working Scripts

* `karate_chop.py` Working script used for development
* `tw_karate_chop.py` working script used to generate visuals

## Graph Generation Methods
HRG offers two methods of generating graphs: 
1.  Exact Generation
2.  Stochastic Genration
    - There is also Probabilistic HRG





## Coding Issues
Status | Description
-------|------------
Closed | Made a mod to `weighted_choice` function; the problem is `n_rhs` (line 432) some times the returned string is a tuple ... check the normal case.

## Notes


## Dev Env

- To switch to the dev version of networkx, make PYTHONPATH `/home/username/local/python/lib/python2.7/site-packages/`

## Referenced work and useful links

* http://stackoverflow.com/questions/24364770/random-sampling-from-a-list-based-on-a-distribution
* [On Assorativity](http://tuvalu.santafe.edu/~aaronc/slides/Clauset_CSSS2014_Networks_3.pdf)
  - [Newman, Phys. Rev. E 67, 026126 (2003).](http://arxiv.org/pdf/cond-mat/0209450v2.pdf)
* [Learning Bayesian Networks from Data |
Nir Friedman Daphne Koller](http://www.cs.huji.ac.il/~nir/Nips01-Tutorial/Nips-tutorial.pdf)

## StarLog

Date    | Notes
--------|------------------------------------------------------------------
19Jun17 | this works: python pami.py --orig ../datasets/out.as20000102

29Jun17 | got phrg, cl, and kp working on small grpahs
29Jun17 | Next: work on larger graphs
02Jul17 | Begin Time experiments and large datasets
04Jul17 | Use the nu metrics to do stats
04Jul17 | working on net stats
04Jul17 | nu_metrix.py pami.py, need to reconsider the CDF
05Jul17 | Working on ECDF merging each result on x "
05Jul17 | Working on ECDF merging each result on x
05Jul17 | Workin on generating dat for large datasets
05Jul17 | ECDF http://ars.els-cdn.com/content/image/1-s2.0-S1877750315300259-gr7.jpg
17Jul17 | ToDo: run both Probs and Rods"
17Jul17 | ToDo: run both Probs and Rods
# PhoenixPython

## StarLog (Version Log)

16Nov16 |`toy.py` reading cindy rules fails in g.sample 
         Walk throug the parsing it might do the thing

09Jan16 |New sampler (in david.py) that should use much much less memory.


## From Tree Decomp to Prod Rules
To print the the Tree:
	in `tree_decomposition`, line 202 (`new_visit`) 
To print the Prod Rules:
	in `tree_decomposition`, line 299 (`add_to_prod_rules`)

## Main Working Scripts

* `karate_chop.py` Working script used for development
* `tw_karate_chop.py` working script used to generate visuals

## Graph Generation Methods
HRG offers two methods of generating graphs: 
1.  Exact Generation
2.  Stochastic Genration
    - There is also Probabilistic HRG





## Coding Issues
Status | Description
-------|------------
Closed | Made a mod to `weighted_choice` function; the problem is `n_rhs` (line 432) some times the returned string is a tuple ... check the normal case.

## Notes


## Dev Env

- To switch to the dev version of networkx, make PYTHONPATH `/home/username/local/python/lib/python2.7/site-packages/`

## Referenced work and useful links

* http://stackoverflow.com/questions/24364770/random-sampling-from-a-list-based-on-a-distribution
* [On Assorativity](http://tuvalu.santafe.edu/~aaronc/slides/Clauset_CSSS2014_Networks_3.pdf)
  - [Newman, Phys. Rev. E 67, 026126 (2003).](http://arxiv.org/pdf/cond-mat/0209450v2.pdf)
* [Learning Bayesian Networks from Data |
Nir Friedman Daphne Koller](http://www.cs.huji.ac.il/~nir/Nips01-Tutorial/Nips-tutorial.pdf)

## StarLog

Date    | Notes
--------|------------------------------------------------------------------
19Jun17 | this works: python pami.py --orig ../datasets/out.as20000102

29Jun17 | got phrg, cl, and kp working on small grpahs
29Jun17 | Next: work on larger graphs
02Jul17 | Begin Time experiments and large datasets
04Jul17 | Use the nu metrics to do stats
04Jul17 | working on net stats
04Jul17 | nu_metrix.py pami.py, need to reconsider the CDF
05Jul17 | Working on ECDF merging each result on x "
05Jul17 | Working on ECDF merging each result on x
05Jul17 | Workin on generating dat for large datasets
05Jul17 | ECDF http://ars.els-cdn.com/content/image/1-s2.0-S1877750315300259-gr7.jpg
17Jul17 | ToDo: run both Probs and Rods"
17Jul17 | ToDo: run both Probs and Rods
25Jul17 | gcd taking for ever
31Aug17 | remote login \ Push URL: git@github.com:abitofalchemy/hrg_nets.git
31Aug17 | working dir: saguinag@dsg1:/data/saguinag/hrg_nets$
01Sep17 | working on exact hrg to save graphs
03Sep17 | set max loops taking a long time!
05Sep17 | got cl_kron_synth.py working to gen pickle files to be used with pgd
