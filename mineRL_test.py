import minerl
import gym
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

def main():
    data = minerl.data.make("MineRLNavigateDenseVectorObf-v0")
    # names = data.get_trajectory_names()
    # print(names[0])
    # stream = data.load_data(names[0])
    # print(stream)
    
    for current_state, action, reward, next_state, done \
    in data.batch_iter(
        batch_size=1, num_epochs=1, seq_len=1):

        #Print the different porperties of the first step of the sequence
        #print(current_state['vector'][0])
        print(action['vector'][0])
        #print(reward[0])
        ##print(next_state['vector'][0])
        #print(done)
        
        
        
if __name__ == '__main__':
    main()