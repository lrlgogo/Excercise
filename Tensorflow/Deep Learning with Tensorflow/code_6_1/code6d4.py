import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[: max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
