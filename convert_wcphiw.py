import numpy as np
testing_dset=np.load("wcwr_l_testing_dataset.npz",allow_pickle=True)
inputs=testing_dset['input']
target=testing_dset['target']
#corrections that need to be executed, expand dimens to 9 or smthing like that
vocab_size=4
size=vocab_size-1
for c in range(len(inputs)):
    i=inputs[c]
    t=target[c]
    for c1 in range(len(i)):
        i0=i[c1] 
        target_new=np.zeros([1,vocab_size*2-1])
        i0_new=np.zeros([1,vocab_size*2-1])
        if c1<len(i)/2-1:
            ind=np.argmax(i0)
            i0_new[0,ind]=1
            target_new=np.ones([1,vocab_size*2-1])
            target_new[0,vocab_size:vocab_size*2-1]=0
        elif c1==int(len(i)/2):
            i0_new[0,0]=1
        else:
            ind=np.argmax(i0)
            i0_new[0,ind+size]=1
        if c1>len(i)/2-1 and c1!=len(i)-1:
            ind0=np.argmax(i[c1+1])
            target_new[0,ind0+size]=1
        inputs[c][c1]=i0_new
        target[c][c1]=target_new
print(inputs[0])
print(target[0])
print(len(target))
np.savez("homo_wcwr_l_testing_dataset.npz",input=inputs,target=target)