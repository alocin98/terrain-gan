import numpy as np
import os

def getData(path='../data/alps_hgt', resolution=128, scale=1, step=0.1, heightDif=1000, midHeight=500):
    data = getHeightModelsFromDirectory(path)
    windows = []
    for heightmodel in data:
        windows.extend(createWindows(heightmodel, resolution, scale, step))
    filtered = list(filter(lambda model: isEnoughHilly(model, heightDif, midHeight), windows))
    filtered = [x / 5000 for x in filtered]
    filtered = numpy.expand_dims(filtered, axis=3)
    return filtered

def getHeightModelsFromDirectory(path):
  filenames = [path + file for file in os.listdir(path)]
  data = []
  for file in filenames:
    siz = os.path.getsize(file)
    dim = int(math.sqrt(siz/2))
    data.append(np.fromfile(file, numpy.dtype('>i2'), dim*dim).reshape((dim, dim)))

  return data

def createWindows(heightmodel, win_size, scale=1, step=0.1):
  windows = []
  for i in range(0,len(heightmodel) - win_size, int(win_size * step)):
    window = heightmodel[i:i+win_size*scale:scale,i:i+win_size*scale:scale];
    windows.append(window)
  return windows

def isEnoughHilly(heightmodel, minDifference = 0, midHeight = 0):
  flattened = np.array(heightmodel).flatten()
  sorted = np.sort(flattened)
  length = len(sorted)
  dif = abs(sorted[0] - sorted[length - 1])
  median = sorted[int(length / 2)]
  return dif > minDifference and median > midHeight and length == 16384

def filterByResolution(heightmodels, resolution):
    filtered = filter(lambda x: len(x) == (resolution * resolution), heightmodels)
    return filtered
