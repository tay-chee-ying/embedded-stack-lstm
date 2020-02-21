import numpy as np
import random
class wcwp_generator:
    def __init__(self,train_lower,train_upper,test_lower,test_upper):
        self.unique_symbols = 2
        self.train_bounds = (train_lower,train_upper)
        self.test_bounds = (test_lower,test_upper)
        return
    def create_input_sequence(self,w_length):
        #create random initial sequence
        sequence = []
        a_count = 0
        string_sentence = ''
        for c in range(w_length):
            index = random.randint(1,self.unique_symbols)
            char = np.zeros([1,self.unique_symbols+1])
            char[0,index] = 1
            sequence += [char]
            if index == 1:
                string_sentence += 'a'
                a_count +=1
            else:
                string_sentence += 'b'
        string_sentence += 'c'
        char = np.zeros([1,self.unique_symbols+1])
        char[0,0] = 1
        sequence +=[char]
        indices = []
        c = 0
        while (len(indices)<a_count):
            index = random.randint(0,w_length-1)
            if not(index in indices):
                indices += [index]
        c = 0
        for c in range(w_length):
            char = np.zeros([1,self.unique_symbols+1])
            if c in indices:
                char[0,1] = 1
                string_sentence += 'a'
            else:
                char[0,2] = 1
                string_sentence += 'b'
            sequence += [char]
        return sequence,string_sentence
    def create_target_sequence(self,sequence):
        w_length =int( (len(sequence) - 1)/2 )
        t_seq = []
        a_count = 0
        for c in range(w_length):
            if sequence[c][0,1] == 1:
                a_count += 1
            t_seq += [np.ones([1,self.unique_symbols+1])]
        #the latter half of the sequence is checking to see if a_count thus far equals to a_count and same for b
        c=0
        na_count = 0
        nb_count = 0
        count = True
        if w_length == 1:
            char = np.zeros([1,3])
            if sequence[0][0,1] == 1:
                char[0,1] = 1
            else:
                char[0,2] = 1
            t_seq+=[char]
        else:
            for c in range(w_length):
                current = sequence[c+w_length]
                char = np.zeros([1,self.unique_symbols+1])
                if count:
                    if current[0,1] == 1:
                        na_count += 1
                    elif current[0,2] == 1:
                        nb_count += 1
                if na_count>=a_count:
                    char[0,2] = 1
                    count = False
                elif nb_count>=(w_length - a_count):
                    char[0,1] = 1
                    count = False
                else:
                    char[0,[1,2]] = 1
                t_seq+=[char]
        t_seq += [np.zeros([1,self.unique_symbols+1])]
        return t_seq
    def dataset_generator(self,sample_num, size, sample_string = []):
        #create n samples 
        sample_in = []
        sample_target = []
        while (len(sample_in)<sample_num):
            w_length = random.randint(int((size[0]-1)/2),int((size[1]-1)/2))
            sentence, s_sentence = self.create_input_sequence(w_length)
            if not(s_sentence in sample_string):
                sample_in+=[sentence]
                sample_target+=[self.create_target_sequence(sentence)]
                sample_string+=[s_sentence]
        return sample_in,sample_target,sample_string
    def train_test_generator(self,train_num,test_num):
        train_in,train_out, sample_string = self.dataset_generator(train_num,self.train_bounds)
        test_in,test_out,s = self.dataset_generator(test_num,self.test_bounds,sample_string)
        np.savez("wcwp_training_dataset.npz",input=train_in,target=train_out)
        np.savez("wcwp_testing_dataset.npz",input=test_in,target=test_out)
        return
if __name__ == "__main__":
    wcwpg = wcwp_generator(1,50,50,100)
    seq,string_sentence = (wcwpg.create_input_sequence(5))
    print(string_sentence)
    tseq = wcwpg.create_target_sequence(seq)
    print(tseq)
    wcwpg.train_test_generator(10000,10000)