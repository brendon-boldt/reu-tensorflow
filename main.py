import numpy as np
import mlp
import prog_bar as bar

ohe_binary = {0:[1,0],1:[0,1]}
#ohe_multi = {1.0:[

def load_data(infile):
  raw_data = np.loadtxt(infile, delimiter=",")
  np.random.shuffle(raw_data)
  x_data = np.array(raw_data[:,:-1])
  #x_data = np.array(raw_data[:,1:])
  y_data = []
  for val in raw_data[:,-1]:
  #for val in raw_data[:,0]:
    if val != 0:
      val = 1
    y_data.append(ohe_binary[val])
  y_data = np.array(y_data)
  return x_data, y_data

def cross_validate(nn, raw_data, folds=5, print_prog=False):
  data_len = raw_data[0].shape[0]
  fold_size = data_len // folds
  rem_size = data_len - (fold_size * (folds-1))

  if not print_prog:
    bar.print_progress(0)
  matrix = {'TP':0,'FP':0,'TN':0,'FN':0}
  for i in range(0,folds-1):
    if print_prog:
      print("\nFold " + str(i+1) + " of " + str(folds))
    start = i*fold_size
    end = start + (folds-1)*fold_size
    x_data = (raw_data[0].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[0].take(range(end, end+fold_size), axis=(0), mode='wrap'))
    y_data = (raw_data[1].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[1].take(range(end, end+fold_size), axis=(0), mode='wrap'))
    new_matrix = nn.run(x_data, y_data, print_prog=print_prog)
    matrix = mlp.MLP.merge_matrices(matrix, new_matrix)
    if not print_prog:
      bar.print_progress(i+1, folds)
    #print("Fold acc:\t" + str(accuracies[-1]/fold_size))
  
  # Final iteratorion
  if print_prog:
    print("\nFold " + str(folds) + " of " + str(folds))
  start = folds*fold_size
  end = start + (folds*fold_size - rem_size)
  x_data = (raw_data[0].take(range(start, end), axis=(0), mode='wrap'),
      raw_data[0].take(range(end, end+rem_size), axis=(0), mode='wrap'))
  y_data = (raw_data[1].take(range(start, end), axis=(0), mode='wrap'),
      raw_data[1].take(range(end, end+rem_size), axis=(0), mode='wrap'))
  new_matrix = nn.run(x_data, y_data, print_prog=print_prog)
  matrix = mlp.MLP.merge_matrices(matrix, new_matrix)
  if not print_prog:
    bar.print_progress(1)
    print()
  #print("Fold acc:\t" + str(accuracies[-1]/rem_size))


  return  matrix

def digest_matrix(m):
  accuracy = (m['TP'] + m['TN']) / sum(m.values())
  print('Accuracy:\t' + "%.3f" % accuracy)
  #precision = m['TP'] / (m['TP'] + m['FP'])
  precision = m['TN'] / (m['TN'] + m['FN'])
  print('Precision:\t' + "%.3f" % precision)
  #recall = m['TP'] / (m['TP'] + m['FN'])
  recall = m['TN'] / (m['TN'] + m['FP'])
  print('Recall:\t\t' + "%.3f" % recall)
  f_measure = 2 * precision * recall / (precision + recall) 
  print('F-measure:\t' + "%.3f" % f_measure)

#data = load_data("tc-b.csv")
#data = load_data("5ns.csv")
data = load_data("ia-b.csv")
input_size = data[0].shape[1]

nn = mlp.MLP()
config = { # ia-b.csv
  'hidden_size':13,
  'iterations':3001,
  'input_size':input_size,
  'num_classes':2,
  'learning_rate':1e-2,
  'batch_size':30,
  'lam':1e2,
  'keep_prob':0.8
}
'''
config = { # R output with 5 number summary - High acc
  'hidden_size':10,
  'iterations':5001,
  'input_size':input_size,
  'num_classes':2,
  'learning_rate':1e-2,
  'batch_size':30,
  'lam':1e3,
  'keep_prob':0.7
}
'''
'''
config = {
  'hidden_size':5,
  'iterations':3001,
  'input_size':input_size,
  'num_classes':2,
  'learning_rate':1e-4,
  'batch_size':30,
  'keep_prob':1.0
}
'''
for var in [0]:
  config = { # ia-b.csv
    'hidden_size':14,
    'iterations': 3001,
    'input_size':input_size,
    'num_classes':2,
    'learning_rate':1e-2,
    'batch_size':40,
    'lam':1e2,
    'keep_prob':0.8
  }
  nn.set_config(config)
  #print('lambda: ' + str(var))
  matrix = cross_validate(nn, data, folds=10, print_prog=False)
  print(matrix)
  digest_matrix(matrix)
  print()
