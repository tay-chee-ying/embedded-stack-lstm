print("START")
import torch
import torch.nn as nn
import numpy as np
import sys
from embedded_stack_LSTM import embedded_stack_lstm_
import time
torch.set_default_tensor_type('torch.DoubleTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
system_number = int(sys.argv[1])
#system_number = 1
random_seed = int("17071701{}".format(str(system_number)))
torch.manual_seed(random_seed)
f = open("/content/drive/My Drive/estack_overdrive/setting_sheet.txt","r")
def get_num(file, typing):
    line = file.readline()
    print(line)
    if typing == "int":
        return int(line[line.find(' ')+1:len(line)-1])
    elif typing == "float":
        return float(line[line.find(' ')+1:len(line)-1])
    else:
        return line[line.find(' ')+1:len(line)-1]
print("--------------")
print("CONFIRMATION: qwerty10")
epsilon = get_num(f,"float")
init_temp = get_num(f,"float")
temp_min = 0.5
ANNEAL_RATE = 0.0000003
hidden_size = get_num(f,"int")
element_size = get_num(f,"int")
embedded_size = get_num(f,"int")
test_embedded_size = get_num(f,"int")
stack_size = get_num(f,"int")
test_stack_size = get_num(f,"int")
learning_rate = get_num(f,"float")
epoch_num = get_num(f,"int")
trials=1
sample_indice=177
dset_name = get_num(f,"name")   
vocab_size = get_num(f,"int")
f.close()
debug_size = 6
print("--------------")
train_name = "/content/drive/My Drive/estack_overdrive/datasets/{}_l_training_dataset.npz".format(dset_name)
test_name = "/content/drive/My Drive/estack_overdrive/datasets/{}_l_testing_dataset.npz".format(dset_name) 
save_prefix = "/content/drive/My Drive/estack_overdrive/save_{}/".format(system_number)
#print to make sure
print("System number: {}".format(system_number))
print("vocab_size: {}, hidden_size: {}, element_size: {}, embedded_size {}, stack_size {}".format(vocab_size,hidden_size,element_size,embedded_size,stack_size))
print("learning_rate: {}, epoch_num: {}, epsilon: {}, init_temp: {}".format(learning_rate,epoch_num,epsilon,init_temp))
print("dset_name: {}".format(dset_name))
print("train_name: {}".format(train_name))
print("test_name: {}".format(test_name))
print("save_prefix: {}".format(save_prefix))
print("------------------------------------")
def target_fix(target):
    target_size=target[0][0].shape
    t_list=[]
    for c in range(len(target)):
        base=torch.zeros([0,target_size[1]]).to(device)
        for c1 in range(len(target[c])):
            target_prime=torch.from_numpy(target[c][c1]).to(device)
            target_prime=target_prime.view(1,-1)
            base=torch.cat((base,target_prime),dim=0)
        t_list+=[base]
    return t_list
def record_decisions(input,model,previous_decisions):
    lstm_hidden,embedded_stack=estack_lstm.init_hidden()
    custom_temp=0.8
    estack_lstm.zero_grad()
    temporal_decisions = np.zeros([1,0,debug_size])
    for c1 in range(len(input)):
        input_symbol=torch.from_numpy(input[c1]).to(device)
        input_symbol=input_symbol.view(1,1,-1)
        output,lstm_hidden,embedded_stack,debug_tuple=estack_lstm(
                input_symbol,lstm_hidden,embedded_stack,custom_temp)
        debug_decisions = debug_tuple.cpu().detach().numpy()
        debug_decisions.shape = (1,1,debug_size)
        temporal_decisions = np.concatenate([temporal_decisions,debug_decisions],axis=1)
    if (previous_decisions.shape)[1] == 0:
        previous_decisions=temporal_decisions
    else:
        previous_decisions = np.concatenate([previous_decisions,temporal_decisions],axis=0)
    return previous_decisions
#get inputs
training_dset = np.load(train_name,allow_pickle = True)
testing_dset = np.load(test_name,allow_pickle = True) 
#init lstm
input = training_dset['input']
target0 = training_dset['target']
target = target_fix(target0)
start_time = time.time()
#forced_desc=training_dset['forced']
for i in range(trials):
    estack_lstm = embedded_stack_lstm_(hidden_size,1,vocab_size,embedded_size,stack_size,element_size).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(estack_lstm.parameters(),lr = learning_rate)
    #optim = torch.optim.SGD(params, lr = learning_rate, momentum = 0.9)
    global_counter = 0
    temp = init_temp
    accumulated_accuracies=[]
    running_average=0
    accumulated_running_average=[]
    accumulated_errors=[]
    previous_decisions = np.zeros([1,0,debug_size])
    for e_c in range(epoch_num):
        for c in range(len(input)):
            optim.zero_grad()
            lstm_hidden,embedded_stack=estack_lstm.init_hidden()
            dummy_output = torch.zeros([len(input[c]),vocab_size]).to(device)
            partial0=0
            decision_target = torch.zeros([len(input[c]),6]).to(device)
            decision_prediction = torch.zeros([len(input[c]),debug_size]).to(device)
            #temp = np.maximum(temp * np.exp(-ANNEAL_RATE), temp_min)
            for c1 in range(len(input[c])):
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE), temp_min)
                #decision scheme is to hold for 2 iters, stack push emb no act, then when # symbol is seen, no-act on both stack and emb and stack2emb. Then pop embedded
                if c1 < 1:
                    decision_target[c1,3] = 1
                elif c1<(len(input[c])-1)/2:
                    decision_target[c1,0] = 1
                elif c1 == (len(input[c])-1)/2:
                    decision_target[c1,5] = 1
                else:
                    decision_target[c1,4] = 1
                input_symbol=torch.from_numpy(input[c][c1]).to(device)
                input_symbol=input_symbol.view(1,1,-1)
                output,lstm_hidden,embedded_stack,debug_tuple=estack_lstm(
                        input_symbol,lstm_hidden,embedded_stack,temp)
                dummy_output[c1,:]=output
                decision_prediction[c1,:] = debug_tuple
                int_output=np.int_(output.cpu().detach().numpy()>epsilon)
                if not(np.all(np.equal(int_output,np.int_(target0[c][c1]>epsilon)))):
                    partial0+=1
            loss = criterion (dummy_output, target[c])
            #loss2 = criterion (decision_prediction, decision_target)
            loss = loss# + loss2
            loss.backward()
            optim.step()
            accumulated_accuracies+=[1-partial0/len(input[c])]
            running_average+=1-partial0/len(input[c])
            accumulated_errors+=[loss.cpu().detach().numpy()]
            # save sample descisions
            if global_counter%1000==0:
                print("Epoch: {} GC: {} Temp: {} MSE Loss: {} Partial Accur: {} Time: {}".format(e_c, global_counter,temp,loss,running_average/10,time.time() - start_time))
                accumulated_running_average+=[running_average/1000]
                start_time = time.time()
                running_average=0
            if global_counter%100==0:
                previous_decisions=record_decisions(input[sample_indice],estack_lstm,previous_decisions)
                np.savetxt(save_prefix+"accumulated_running_average.csv",np.array(accumulated_running_average),delimiter=',')
                np.savetxt(save_prefix+"accumulated_accuracies.csv",np.array(accumulated_accuracies),delimiter=',')
                np.savetxt(save_prefix+"accumulated_errors.csv",np.array(accumulated_errors),delimiter=',')
            if global_counter%500==0:
                np.savez(save_prefix+"in_training_decisions.npz",decisions=previous_decisions)
                torch.save(estack_lstm.state_dict(), save_prefix+"estack_0.pt")
            global_counter+=1
#perform testing
print("-----------------------")
def test_model(estack_lstm, testing_dset):
    #iterate over all samples
    input=testing_dset['input']
    target=testing_dset['target']
    total_count=0
    total_partials=0
    with torch.no_grad():
        iters=len(target)
        mae=0
        for c in range(iters):
            if c%1000==0:
                print(c,"/",iters," complete")
            sample_length=len(input[c])
            estack_lstm.zero_grad()
            #build estack hidden inits
            lstm_hidden,embedded_stack=estack_lstm.init_hidden()
            #lstm_hidden,prev_head,embedded_stack,hc=(lstm_hidden[0].to(device),lstm_hidden[1].to(device)),prev_head.to(device),embedded_stack.to(device),hc.to(device)
            success=1
            partials=0
            v0=np.zeros([0,vocab_size])
            mae_partial=0
            for c1 in range(sample_length):
                input_symbol=torch.from_numpy(input[c][c1]).to(device)
                input_symbol=input_symbol.view(1,1,-1)
                output,lstm_hidden,embedded_stack,debug=estack_lstm(
                        input_symbol,lstm_hidden,embedded_stack,0.5)
                int_output=np.int_(output.cpu().detach().numpy()>epsilon)
                np_out=output.cpu().detach().numpy()
                np_out.shape=[1,vocab_size]
                v0=np.concatenate([v0,np_out],axis=0)
                if not(np.all(np.equal(int_output,np.int_(target[c][c1]>epsilon)))):
                    success=0
                    partials+=1
                mae_partial+=np.mean(np.abs(np_out-target[c][c1]))
            total_count+=success
            total_partials+=(1-float(partials)/sample_length)
            mae+=mae_partial/sample_length
    #save output sample
    return 100.0*float(total_count)/iters,100.0*total_partials/iters,mae/iters
test_estack = embedded_stack_lstm_(hidden_size,1,vocab_size,test_embedded_size,test_stack_size,element_size)
test_estack.load_state_dict(torch.load(save_prefix+"estack_0.pt"))
test_estack.zero_grad()
train_total_accur, train_partial_accur, train_mae = test_model(test_estack, training_dset)
test_total_accur, test_partial_accur, test_mae = test_model(test_estack, testing_dset)
results = np.concatenate([np.array([[train_total_accur,train_partial_accur,train_mae]]),np.array([[test_total_accur,test_partial_accur,test_mae]])],axis=1)
np.savetxt(save_prefix + "eval_results.csv",results,delimiter=',')