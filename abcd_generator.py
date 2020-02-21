import numpy as np
import random
class abcd_generator:
    def __init__(self,train_lower,train_upper,test_lower,test_upper):
        self.unique_symbols = 2
        self.train_bounds = (train_lower,train_upper)
        self.test_bounds = (test_lower,test_upper)
        return
    def create_input_sequence(self,n_length):
        a_char = np.zeros([1,4])
        b_char = np.zeros([1,4])
        c_char = np.zeros([1,4])
        d_char = np.zeros([1,4])
        a_char[0,0] = 1
        b_char[0,1] = 1
        c_char[0,2] = 1
        d_char[0,3] = 1
        sequence = [a_char]*n_length + [b_char]*n_length + [c_char]*n_length + [d_char]*n_length
        string_seq = 'a'*n_length + 'b'*n_length + 'c'*n_length + 'd'*n_length
        return sequence, string_seq
    def create_target_sequence(self,sequence):
        n_length = int(len(sequence)/4)
        ab = np.zeros([1,4])
        b = np.zeros([1,4])
        c = np.zeros([1,4])
        d = np.zeros([1,4])
        ab[0,[0,1]] = 1
        b[0,1] = 1
        c[0,2] = 1
        d[0,3] = 1
        tseq = []
        tseq += [ab]*n_length
        tseq += [b]*(n_length - 1)
        tseq += [c]*(n_length)
        tseq += [d]*(n_length)
        tseq += [np.zeros([1,4])]
        return tseq
    def dataset_generator(self,sample_num, size, sample_string = []):
        #create n samples 
        sample_in = []
        sample_target = []
        while (len(sample_in)<sample_num):
            n_length = random.randint(int((size[0])/4),int((size[1])/4))
            sentence, s_sentence = self.create_input_sequence(n_length)
            if not(s_sentence in sample_string):
                sample_in+=[sentence]
                sample_target+=[self.create_target_sequence(sentence)]
                sample_string+=[s_sentence]
        return sample_in,sample_target,sample_string
    def train_test_generator(self,train_num,test_num):
        train_in,train_out, sample_string = self.dataset_generator(train_num,self.train_bounds)
        test_in,test_out,s = self.dataset_generator(test_num,self.test_bounds,sample_string)
        np.savez("abcd_training_dataset.npz",input=train_in,target=train_out)
        np.savez("abcd_testing_dataset.npz",input=test_in,target=test_out)
        return
a = abcd_generator(1,100,100,200)
seq, sseq = a.create_input_sequence(3)
print(sseq)
print(a.create_target_sequence(seq))
a.train_test_generator(15,15)