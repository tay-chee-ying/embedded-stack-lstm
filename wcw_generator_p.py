import numpy as np
import random

class wcw_language():
    def __init__(self, alphabet_size,min_size,max_size):
        self.alpha_size=alphabet_size
        self.size=(min_size,max_size)
        #init vocab, note max size is 93 unique characters bcos of ascii
        self.vocab=[]
        for c in range(alphabet_size):
            self.vocab+=[chr(33+c)]
        return
    def create_input_sequence(self,sizes):
        current_vocab=(self.vocab).copy()
        #select=random.randint(0,len(current_vocab)-1)
        select=0
        separator = current_vocab.pop(select)
        input_seq=''
        for c in range(random.randint(int(sizes[0]/2)+1,int(sizes[1]/2)+1)):
            input_seq+=random.choice(current_vocab)
        return input_seq+separator+input_seq
    def tokenize_input_sequence(self,input_str):
        #create numpy array with dimens (1, alpha_size)
        tokenized=[]
        for c in range(len(input_str)):
            one_hot=np.zeros([1,self.alpha_size])
            one_hot[0,ord(input_str[c])-33]=1
            tokenized+=[one_hot]
        return tokenized
    def create_output_sequence(self,input_str,token_input):
        #any symbol is possible from the first symbol up to symbol before c
        all_possible=np.ones([1,self.alpha_size])
        proto_target=[all_possible]*int((len(input_str)-1)/2)
        back=token_input[0:int((len(input_str)-1)/2)]
        terminator=[np.zeros([1,self.alpha_size])]
        return proto_target+back+terminator
    def forced_descisions(self,input_str):
        p1_desc=np.array([[1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0]])
        p2_desc=np.array([[0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]])
        p3_desc=np.array([[0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0]])
        desc_target=[p1_desc]*int((len(input_str)-1)/2)+[p2_desc]*(int((len(input_str)-1)/2))+[p3_desc]
        return desc_target
    def dataset_generator(self,number):
        input_dset=[]
        target_dset=[]
        descision_dset=[]
        str_dset=[]
        not_done=True
        while not_done:
            if len(str_dset)>=10000:
                size=[30,50]
            else:
                size=self.size
            i=self.create_input_sequence(size)
            li=self.tokenize_input_sequence(i)
            os=self.create_output_sequence(i,li)
            dt=self.forced_descisions(i)
            if not(i in str_dset):
                str_dset+=[i]
                descision_dset+=[dt]
                input_dset+=[li]
                target_dset+=[os]
            if len(str_dset)==number:
                not_done=False
        return input_dset,target_dset,descision_dset
    def test_train_split(self,train_size,test_size):
        input_dset,target_dset,dds=self.dataset_generator(train_size+test_size)
        length=len(input_dset)
        return input_dset[0:train_size],target_dset[0:train_size],input_dset[train_size-1:length-1],target_dset[train_size-1:length-1]
language=wcw_language(4,10,30)
training_ids,training_tds,testing_ids,testing_tds=language.test_train_split(10000,10000)
print(language.create_input_sequence([10,20]))
print(len(training_ids),len(testing_tds))
np.savez("wcw_l_training_dataset.npz",input=training_ids,target=training_tds)
np.savez("wcw_l_testing_dataset.npz",input=testing_ids,target=testing_tds)