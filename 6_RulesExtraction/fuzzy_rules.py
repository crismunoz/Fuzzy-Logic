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

def config_input_variable(name, ns, min_val, max_val, resolution, shoulder):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Antecedent(universe, name)
  var = config_set_variable(var, ns, min_val, max_val, shoulder)
  return var

def config_output_variable(name, ns, min_val, max_val, resolution, shoulder, defuzzify_method):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Consequent(universe, name, defuzzify_method=defuzzify_method)
  var = config_set_variable(var, ns, min_val, max_val, shoulder)
  return var

def config_set_variable(var, ns, min_val, max_val, shoulder):
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

def define_input_variables(config, shoulder):
  epsilon = config['epsilon']
  input_variables  = [config_input_variable('I_{}'.format(i+1), 
  config['nb_sets'], 
  config['min'][0][i]-epsilon, 
  config['max'][0][i]+epsilon, 
  config['resolution'], 
  shoulder=shoulder) \
                    for i in range(config['nb_inputs'])]
  return input_variables

def define_output_variables(config, shoulder, defuzzify_method):
  epsilon = config['epsilon']
  output_variables = [config_output_variable('O_{}'.format(i+1), 
  config['nb_sets'], 
  config['min'][1][i]-epsilon, 
  config['max'][1][i]+epsilon, 
  config['resolution'], 
  shoulder=shoulder, 
  defuzzify_method=defuzzify_method) \
                    for i in range(config['nb_outputs'])]
  return output_variables

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
    us = [u for t,u in rule_candidate]
    dr  = config['intersection_op'](us) 
    key = '_'.join([str(t) for t,u in rule_candidate[:-1]]) 
    all_rule_candidate.append((key, dr, rule_candidate))
  
  sorted_all_rule_candidate = sorted(all_rule_candidate, key=itemgetter(0))
  groups = groupby(sorted_all_rule_candidate, key=itemgetter(0))
  ant_con = []
  for k,v in groups:
    _,dr,rule_candidate = zip(*v)
    choosen = np.argmax(dr)
    rule = rule_candidate[choosen]
    dr_val = dr[choosen]
    terms,_ = zip(*rule)
    ant = terms[:-1]
    con = terms[-1]
    ant_con.append((ant,con, dr_val))

  rules = []
  rules_view = []
  for ant,con,dr_val in ant_con:
    ant_vars = []
    for a,var in zip(ant,input_variables):
      terms = [t for t in var.terms]
      ant_vars.append(var[terms[a]])
    rule_str=' &  '.join(['ant_vars[{}]'.format(i) for i in range(len(ant_vars))])
    antc = eval(rule_str)

    var = output_variables[-1]
    terms = [t for t in var.terms]
    cons = var[terms[con]]
    rules.append(ctrl.Rule(antc, cons, **config['aggregation_opt']))
    rules_view.append((antc, cons, dr_val))
  return rules, rules_dataframe(rules_view)

def rules_dataframe(rules_view):
  df=pd.DataFrame()
  for rule in rules_view:
      dr_val = rule[2]
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
      order_columns = ['I_{}'.format(i+1) for i,a in enumerate(ts[::-1])] + ['O_1', 'Dr']
      rule_dict = {'I_{}'.format(i+1):a.label for i,a in enumerate(ts[::-1])}
      rule_dict['O_1']= con.label
      rule_dict['Dr']= dr_val
      df=df.append(rule_dict, ignore_index=True)
      df = df[order_columns]
  return df
