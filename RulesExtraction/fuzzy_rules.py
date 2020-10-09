from operator import itemgetter
from itertools import groupby
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from functools import reduce

def config_input_variable(name, ns, min, max, resolution=500):
  universe = np.linspace(min,max, resolution)
  var = ctrl.Antecedent(universe, name)
  var = config_variable(var, ns, min, max)
  return var

def config_output_variable(name, ns, min, max, resolution=500):
  universe = np.linspace(min,max, resolution)
  var = ctrl.Consequent(universe, name)
  var = config_variable(var, ns, min, max)
  return var

def config_variable(var, ns, min, max):
  c = np.linspace(min,max, ns+2)
  d = ( max - min ) / (ns - 1)
  var['s_1'] = fuzz.trapmf(var.universe, [c[1] - 2*d, c[1] -d , c[1], c[1] + d])
  for s in range(2,ns):
    var['s_{}'.format(s)] = fuzz.trimf(var.universe, [c[s] -d, c[s], c[s] + d])
    var['s_{}'.format(ns)] = fuzz.trapmf(var.universe, [c[ns] -d , c[ns], c[ns] + d, c[ns] + 2*d])
  return var

def extract_rules(config, X, y):
  data = np.concatenate([X,y],axis=1)
  input_variables  = [config_input_variable('i_{}'.format(i+1), config['nb_sets'], config['min'], config['max'], config['resolution']) \
                    for i in range(config['nb_inputs'])]
  output_variables = [config_output_variable('o_{}'.format(i+1), config['nb_sets'], config['min'], config['max'], config['resolution']) \
                    for i in range(config['nb_outputs'])]
    
  variables = input_variables + output_variables

  all_rule_candidate = []
  for instance in data:
    rule_candidate = []
    for x,var in zip(instance,variables):
      u_x_s = [fuzz.interp_membership(var.universe, var[term].mf, x)  for term in var.terms]
      term_max = np.argmax(u_x_s)
      rule_candidate.append((term_max , u_x_s[term_max]))
    dr  = np.prod([u for t,u in rule_candidate]) 
    key = '_'.join([str(t) for t,u in rule_candidate[:-1]]) 
    all_rule_candidate.append((key, dr, rule_candidate))
  
  sorted_all_rule_candidate = sorted(all_rule_candidate, key=itemgetter(0))
  groups = groupby(sorted_all_rule_candidate, key=itemgetter(0))
  ant_con = []
  for k,v in groups:
    _,dr,rule_candidate = zip(*v)
    choosen = np.argmax(dr)
    rule = rule_candidate[choosen]
    terms,_ = zip(*rule)
    ant = terms[:-1]
    con = terms[-1]
    ant_con.append((ant,con))

  rules = []
  for ant,con in ant_con:
    ant_vars = []
    for a,var in zip(ant,input_variables):
      terms = [t for t in var.terms]
      ant_vars.append(var[terms[a]])
    antc = reduce(lambda x,y:x & y, ant_vars)

    var = output_variables[-1]
    terms = [t for t in var.terms]
    cons = var[terms[con]]
    rules.append(ctrl.Rule(antc, cons))

  return rules 