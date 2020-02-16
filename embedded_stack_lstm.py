import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class embedded_stack_lstm_(nn.Module):
    def __init__(self, hidden_size,n_layers,output_size,embedded_size,stack_size,element_size):
        super(embedded_stack_lstm_,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.M_size=(embedded_size,stack_size,element_size)
        self.lstm = nn.LSTM(output_size, hidden_size, n_layers)
        self.plinear = nn.Linear(element_size,hidden_size)
        self.emblinear = nn.Linear(hidden_size,3)
        self.curlinear = nn.Linear(hidden_size,3)
        self.selectlinear = nn.Linear(hidden_size,3)
        self.olinear = nn.Linear(hidden_size,output_size)
        self.esymblinear = nn.Linear(hidden_size,element_size)
        self.ssymblinear = nn.Linear(hidden_size,element_size)
        self.sigmoid = nn.Sigmoid ()
        self.tanh = nn.Tanh()
        return
    def init_hidden(self):
        embedded_stack=torch.zeros(self.M_size).to(device)
        lstm_hidden=(torch.zeros (self.n_layers, 1, self.hidden_size).to(device),
                torch.zeros (self.n_layers, 1, self.hidden_size).to(device))
        previous_head=torch.zeros(self.M_size[0]).to(device)
        previous_head[0]=1
        return lstm_hidden,previous_head,embedded_stack
    def forward(self, input, hidden0, previous_head, embedded_stack,temperature):
        if type(temperature) != tuple:
            temperature = (temperature, temperature, temperature)
        output, (ht, ct) = self.lstm(input, hidden0)
        d_select = F.gumbel_softmax(self.selectlinear(output.view(-1)),tau=temperature[2])
        
        read = (embedded_stack[:,0,:]*previous_head.view(-1,1).repeat(1,self.M_size[2])).sum(dim=0)
        ht = self.tanh(ht + self.tanh(self.plinear(read.view(1, -1)).view(1, 1, -1)))
        decision_input = output.view(-1)
        d_emb,d_cur = F.gumbel_softmax(self.emblinear(decision_input),tau=temperature[0]),F.gumbel_softmax(self.curlinear(decision_input),tau=temperature[1])
        y = self.olinear(output.view(-1))
        
        estack_symb = torch.zeros([1,self.M_size[1],self.M_size[2]]).to(device)
        estack_symb[0,0,:] = self.sigmoid(self.esymblinear(ht)).view(-1)
        stack_symb = self.sigmoid(self.esymblinear(ht)).view(1,1,-1).repeat(self.M_size[0],1,1)
        
        emb_push = torch.cat([estack_symb,embedded_stack[0:self.M_size[0]-1,:,:]],0)
        emb_pop = torch.cat([embedded_stack[1:self.M_size[0],:,:],torch.zeros([1,self.M_size[1],self.M_size[2]]).to(device)],0)
        embedded_stack_1 = emb_push*d_emb[0] + embedded_stack*d_emb[1] + emb_pop*d_emb[2]
        #embedded_stack_1 = embedded_stack
        stack_push = torch.cat([stack_symb,embedded_stack_1[:,0:self.M_size[1]-1,:]],1)
        stack_pop = torch.cat([embedded_stack_1[:,1:self.M_size[1],:],torch.zeros([self.M_size[0],1,self.M_size[2]]).to(device)],1)
        embedded_stack_2 = stack_push*d_cur[0] + embedded_stack_1*d_cur[1] + stack_pop*d_cur[2]
        new_embedded_stack = embedded_stack_1 * (1 - previous_head.view(-1,1,1).repeat([1,self.M_size[1],self.M_size[2]])) + embedded_stack_2 * previous_head.view(-1,1,1).repeat([1,self.M_size[1],self.M_size[2]])
        
        shift_right = torch.cat([torch.zeros([1]).to(device),previous_head[0:self.M_size[0]-1]],0) 
        shift_left = torch.cat([previous_head[1:self.M_size[0]],torch.zeros([1]).to(device)],0)
        next_head = shift_right*d_select[0] + previous_head*d_select[1] + shift_left*d_select[2]
        next_head = next_head/next_head.sum()
        
        #next_read = (new_embedded_stack[:,0,:]*next_head.view(-1,1).repeat(1,self.M_size[2])).sum(dim=0)
        #ct = self.tanh(ct + self.tanh(self.plinear(next_read.view(1, -1)).view(1, 1, -1)))
        
        debug_tuple=torch.cat([d_emb,d_cur,d_select],0)
        return y, (ht.view(1,1,-1),ct), next_head,new_embedded_stack,debug_tuple