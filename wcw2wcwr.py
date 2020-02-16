import numpy as np
testing_dset=np.load("wcw_l_testing_dataset.npz",allow_pickle=True)
inputs=testing_dset['input']
target=testing_dset['target']
print(inputs[0])
print("----------")
print(target[0])
print("=======================")
for c in range(len(inputs)):
    data = inputs[c]
    target_data = target[c]
    d_len = len(data)
    for c1 in range(int((d_len-1)/2)):
        data[d_len-c1-1] = data[c1]
        target_data[d_len - c1 -2] = data[c1]
    inputs[c] = data
    target[c] = target_data
print(inputs[0])
print("--------")
print(target[0])
np.savez("wcwr_l_testing_dataset.npz",input=inputs,target=target)