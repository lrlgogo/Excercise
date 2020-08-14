import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))

state_t = np.zeros((output_features,))
print('initial state:')
print(state_t)

W = np.random.normal(size=(output_features, input_features))
U = np.random.normal(size=(output_features, output_features))
b = np.random.normal(size=(output_features,))

print('W = ')
print(W)
print('U = ')
print(U)
print('b = ')
print(b)

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
    print(state_t)

final_output_sequence = np.stack(successive_outputs, axis=0)

print('final output sequence = ')
print(final_output_sequence)
