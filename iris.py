import numpy as np
import mlp

labels = {b"Iris-setosa":0, b"Iris-versicolor":1, b"Iris-virginica":2}

def load_data(infile):
  raw_data = np.loadtxt(infile, delimiter=",", converters={4:lambda x: labels[x]})
  np.random.shuffle(raw_data)
  x_data = np.array(raw_data[:,:4])
  y_data = []
  for val in raw_data[:,4]:
    if val == 0.0:
      y_data.append([1,0,0])
    elif val == 1.0:
      y_data.append([0,1,0])
    elif val == 2.0:
      y_data.append([0,0,1])
  y_data = np.array(y_data)
  return x_data, y_data

def cross_validate(nn, raw_data, folds=5):
  data_len = raw_data[0].shape[0]
  fold_size = data_len // folds
  rem_size = data_len - (fold_size * folds)

  accuracies = []
  for i in range(0,folds):
    start = i*fold_size
    end = start + (folds-1)*fold_size
    x_data = (raw_data[0].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[0].take(range(end, end+fold_size), axis=(0), mode='wrap'))
    y_data = (raw_data[1].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[1].take(range(end, end+fold_size), axis=(0), mode='wrap'))
    accuracies.append(nn.run(x_data, y_data, print_prog=False)*fold_size)
    #print("Fold acc:\t" + str(accuracies[-1]/fold_size))
  if rem_size != 0:
    start = folds*fold_size
    end = start + (folds*fold_size - rem_size)
    x_data = (raw_data[0].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[0].take(range(end, end+rem_size), axis=(0), mode='wrap'))
    y_data = (raw_data[1].take(range(start, end), axis=(0), mode='wrap'),
        raw_data[1].take(range(end, end+rem_size), axis=(0), mode='wrap'))
    accuracies.append(nn.run(x_data, y_data, print_prog=True) * rem_size)
    #print("Fold acc:\t" + str(accuracies[-1]/rem_size))

  return accuracies, data_len

nn = mlp.MLP()
config = {
  'hidden_size':3,
  'iterations':10000,
  'input_size':4,
  'num_classes':3
}
nn.set_config(config)
accs, data_size = cross_validate(nn, load_data("iris.csv"), folds=4)
#print(accs)
print(sum(accs)/data_size)

