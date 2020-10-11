from operator import itemgetter
from itertools import groupby
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from functools import reduce
import pandas as pd

def read_dataset(file_path):
    def preprocessing(line):
        values = line.strip().replace(',','.').split('\t')
        return [float(v) for v in values]

    with open(file_path,'r') as file:
        series = [preprocessing(line) for line in file]
    return np.array(series)

def config_input_variable(name, ns, min_val, max_val, resolution=500, shoulder=True):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Antecedent(universe, name)
  var = config_variable(var, ns, min_val, max_val, shoulder)
  return var

def config_output_variable(name, ns, min_val, max_val, resolution=500, shoulder=True):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Consequent(universe, name)
  var = config_variable(var, ns, min_val, max_val, shoulder)
  return var

def config_variable(var, ns, min_val, max_val, shoulder=True):
  if shoulder:
      c = np.linspace(min_val, max_val, ns+2)
      d = ( max_val - min_val ) / (ns + 1)
      var['s_1'] = fuzz.trapmf(var.universe, [c[0]-d, c[0] , c[1], c[2]])
      for s in range(2,ns):
        var['s_{}'.format(s)] = fuzz.trimf(var.universe, [c[s-1], c[s], c[s+1]])
      var['s_{}'.format(ns)] = fuzz.trapmf(var.universe, [c[ns-1] , c[ns], c[ns+1], c[ns] + d])
      return var
  else:
      c = np.linspace(min_val, max_val, ns)
      d = ( max_val - min_val ) / (ns - 1)
      for s in range(1,ns+1):
        var['s_{}'.format(s)] = fuzz.trimf(var.universe, [c[s-1] -d, c[s-1], c[s-1] + d])
      return var

def define_variables(config, shoulder):
  input_variables  = [config_input_variable('i_{}'.format(i+1), config['nb_sets'], 0.999*config['min'][0][i], 1.001*config['max'][0][i], config['resolution'], shoulder=shoulder) \
                    for i in range(config['nb_inputs'])]
  output_variables = [config_output_variable('o_{}'.format(i+1), config['nb_sets'], 0.999*config['min'][1][i], 1.001*config['max'][1][i], config['resolution'], shoulder=shoulder) \
                    for i in range(config['nb_outputs'])]
  return input_variables,output_variables

def extract_rules(config, input_variables , output_variables, X, y):
  variables = input_variables + output_variables
  data = np.concatenate([X,y],axis=1)
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
  rules_view = []
  for ant,con in ant_con:
    ant_vars = []
    for a,var in zip(ant,input_variables):
      terms = [t for t in var.terms]
      ant_vars.append(var[terms[a]])
    rule_str=' &  '.join(['ant_vars[{}]'.format(i) for i in range(len(ant_vars))])
    antc = eval(rule_str)

    var = output_variables[-1]
    terms = [t for t in var.terms]
    cons = var[terms[con]]
    rules.append(ctrl.Rule(antc, cons))
    rules_view.append((antc, cons))
  return rules, rules_dataframe(rules_view)

def rules_dataframe(rules_view):
  df=pd.DataFrame()
  for rule in rules_view:
      con = rule[1]
      t1=rule[0]
      ts=[]
      while True:
          try:
              t2=t1.term2
              ts.append(t2)
              t1=t1.term1        
          except:
              ts.append(t1)
              break
      rule_dict = {'i_{}'.format(i+1):a.label for i,a in enumerate(ts[::-1])}
      rule_dict['o_1']= con.label
      df=df.append(rule_dict, ignore_index=True)
  return df
