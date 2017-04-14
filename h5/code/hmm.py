#!/usr/bin/env python 

from util import * 
from numpy import *
# from math import log
import copy
import sys


# Pretty printing for 1D/2D numpy arrays
MAX_PRINTING_SIZE = 30

def format_array(arr):
    s = shape(arr)
    if s[0] > MAX_PRINTING_SIZE or (len(s) == 2 and s[1] > MAX_PRINTING_SIZE):
        return "[  too many values (%s)   ]" % s

    if len(s) == 1:
        return  "[  " + (
            " ".join(["%.6f" % float(arr[i]) for i in range(s[0])])) + "  ]"
    else:
        lines = []
        for i in range(s[0]):
            lines.append("[  " + "  ".join(["%.6f" % float(arr[i,j]) for j in range(s[1])]) + "  ]")
        return "\n".join(lines)



def format_array_print(arr):
    print(format_array(arr))


def string_of_model(model, label):
    (initial, tran_model, obs_model) = model
    return """
Model: %s 
initial: 
%s

transition: 
%s

observation: 
%s
""" % (label, 
       format_array(initial),
       format_array(tran_model),
       format_array(obs_model))

    
def check_model(model):
    """Check that things add to one as they should"""
    (initial, tran_model, obs_model) = model
    for state in range(len(initial)):
        assert((abs(sum(tran_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(obs_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(initial) - 1)) <= 0.01)


def print_model(model, label):
    check_model(model)
    print(string_of_model(model, label))

def max_delta(model, new_model):
    """Return the largest difference between any two corresponding 
    values in the models"""
    return max( [(abs(model[i] - new_model[i])).max() for i in range(len(model))] )


class HMM:
    """ HMM Class that defines the parameters for HMM """
    def __init__(self, states, outputs):
        """If the hmm is going to be trained from data with labeled states,
        states should be a list of the state names.  If the HMM is
        going to trained using EM, states can just be range(num_states)."""
        self.states = states
        self.outputs = outputs
        n_s = len(states)
        n_o = len(outputs)
        self.num_states = n_s
        self.num_outputs = n_o
        self.initial = zeros(n_s)
        self.transition = zeros([n_s,n_s])
        self.observation = zeros([n_s, n_o])

    def set_hidden_model(self, init, trans, observ):
        """ Debugging function: set the model parameters explicitly """
        self.num_states = len(init)
        self.num_outputs = len(observ[0])
        self.initial = array(init)
        self.transition = array(trans)
        self.observation = array(observ)
        
    def get_model(self):
        return (self.initial, self.transition, self.observation)

    def compute_logs(self):
        """Compute and store the logs of the model (helper)"""
        raise Exception("Not implemented")

    def __repr__(self):
        return """states = %s
observations = %s
%s
""" % (" ".join(array_to_string(self.states)), 
       " ".join(array_to_string(self.outputs)), 
       string_of_model((self.initial, self.transition, self.observation), ""))

     
    # declare the @ decorator just before the function, invokes print_timing()
    # @print_timing
    def learn_from_labeled_data(self, state_seqs, obs_seqs):

        theta_hat = zeros(self.num_states)

        for i in range(0,self.num_states):
            summer = 1.0
            for j in state_seqs:
                if(i==j[0]):
                    summer+=1

            # print("HERE BE SUMMER",summer)
            # print("HERE BE self.num_states",self.num_states)
            # print("HERE BE state_seqs",len(state_seqs))
            theta_hat[i]=(summer/(self.num_states+len(state_seqs)))
            # print("HERE IS THETAHAT SUBSET",theta_hat[i])
            # print("YO HERE's 1. by 6",1.0/6)



        t_hat = ones((self.num_states,self.num_states))
        count = self.num_states*ones(self.num_states)

        for i in state_seqs:
            for j in range(0,len(i)-1):
                t_hat[i[j]][i[j+1]]+=1
                count[i[j]]+=1

        for i in range(0,t_hat.shape[0]):
            t_hat[i]=divide(t_hat[i],count)



        pi_hat = ones((self.num_states,self.num_outputs))
        count2 = self.num_states*ones(self.num_states)

        for i in range(0,len(state_seqs)):
            for j in range(0,len(state_seqs[i])):
                pi_hat[state_seqs[i][j]][obs_seqs[i][j]]+=1
                count2[state_seqs[i][j]]+=1

        for i in range(0,pi_hat.shape[1]):
            pi_hat[:,i]=divide(pi_hat[:,i],count2)
            # print("HIIIII")
            # print(pi_hat[:,i])

        # print("HSDSFDF ITS THETA HAT",theta_hat)
        # print("PI HAT IS", pi_hat)
        self.initial = theta_hat
        self.transition = t_hat
        self.observation = pi_hat


        """
        Learn the parameters given state and observations sequences. 
        The ordering of states in states[i][j] must correspond with observations[i][j].
        Use Laplacian smoothing to avoid zero probabilities.
        Implement for (a).
        """

        # Fill this in...
        #raise Exception("Not implemented")
        

    def most_likely_states(self, sequence, debug=True):
        """Return the most like sequence of states given an output sequence.
        Uses Viterbi algorithm to compute this.
        Implement for (b) and (c).
        """
        # Fill this in...
        t1=zeros((len(sequence),self.num_states))
        t2=zeros((len(sequence),self.num_states))
        
        for i in range(0,self.num_states):
            # self.observation[i][sequence[0]]
            #t1[0][i] = self.initial[i]*self.observation[i][sequence[0]]
            t1[0][i] = log(self.initial[i]*self.observation[i][sequence[0]])
            print(t1[0][i])
            t2[0][i] = 0

        for i in range(1,len(sequence)):
            for j in range(0,self.num_states):
                #t1[i,j] = self.observation[j,sequence[i]]*max(t1[i-1,:]*self.transition[:,j])
                # print("is array?", self.observation[j,sequence[i]]*self.transition[:,j])
                # print("will this print", log(self.observation[j,sequence[i]]*self.transition[:,j]))
                t1[i,j] = max(log(self.observation[j,sequence[i]]*self.transition[:,j])+t1[i-1,:])
                #t2[i,j] = argmax(t1[i-1,:]*self.transition[:,j])
                t2[i,j] = argmax(t1[i-1,:]+log(self.transition[:,j]))

        print(t1[2])

        statesequences = zeros(len(sequence))
        # print(z.shape)
        x = zeros(len(sequence))
        # print("len sequence",len(sequence))
        # print("TIS x",x)
        # print(x.shape)

        for i in range(len(sequence)-1,-1,-1):
            if i == len(sequence)-1:
                statesequences[i] = int(argmax(t1[-1,:]))
                # print("statesequences here!",statesequences[len(sequence)-1])
                # print(t1[len(sequence)-1,:])
                #x[len(sequence)-1] = t2[len(sequence)-1,statesequences[len(sequence)-1]]
                x = t2[i,argmax(t1[-1,:])]
            else:
                statesequences[i] = int(x)
                #statesequences[i] = x[i+1]
                #t2[i,int(statesequences[i+1])]
                #print(statesequences[i])
                x = t2[i,int(x)]
                #x[i] = t2[i][int(x[i+1])]
                #statesequences[i]

        # print("ABOUT TO DO THIS FDSIOSDOPJF")
        # print(x)
        # print(len(x))
        x = x.tolist()
        statesequences = statesequences.tolist()
        final = zeros(len(sequence),dtype=int)
        print("FINAL",final)
        for i in range(0,len(statesequences)):
            final[i] = int(statesequences[i])
        print("FINAL",final)
        # print("VS",statesequences)
        # print("int test",int(statesequences[0]))
        # print("TIS stateseqs",int(statesequences))
        return final
    
def get_wikipedia_model():
    # From the rainy/sunny example on wikipedia (viterbi page)
    hmm = HMM(['Rainy','Sunny'], ['walk','shop','clean'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.4,0.5], [0.6,0.3,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

def get_toy_model():
    hmm = HMM(['h1','h2'], ['A','B'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.9], [0.9,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm
    

