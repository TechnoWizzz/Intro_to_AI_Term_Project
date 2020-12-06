class agent():
      def __init__(self, gamma=0.99, replace=100, lr=0.001):
          self.gamma = gamma
          self.epsilon = 1.0
          self.min_epsilon = 0.01
          self.epsilon_decay = 1e-3
          self.replace = replace
          self.trainstep = 0
          self.memory = exp_replay()
          self.batch_size = 64
          self.q_net = DDDQN()
          self.target_net = DDDQN()
          opt = tf.keras.optimizers.Adam(learning_rate=lr)
          self.q_net.compile(loss='mse', optimizer=opt)
          self.target_net.compile(loss='mse', optimizer=opt)
      
      
      def act(self, state):
          if np.random.rand() <= self.epsilon:
              return np.random.choice([i for i in range(env.action_space.n)])

          else:
              actions = self.q_net.advantage(np.array([state]))
              action = np.argmax(actions)
              return action


      
      def update_mem(self, state, action, reward, next_state, done):
          self.memory.add_exp(state, action, reward, next_state, done)


      def update_target(self):
          self.target_net.set_weights(self.q_net.get_weights())     

      def update_epsilon(self):
          self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
          return self.epsilon
          
            
      def train(self):
          if self.memory.pointer < self.batch_size:
             return 
          
          if self.trainstep % self.replace == 0:
             self.update_target()
          states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
          target = self.q_net.predict(states)
          next_state_val = self.target_net.predict(next_states)
          max_action = np.argmax(self.q_net.predict(next_states), axis=1)
          batch_index = np.arange(self.batch_size, dtype=np.int32)
          q_target = np.copy(target)  #optional  
          q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
          self.q_net.train_on_batch(states, q_target)
          self.update_epsilon()
          self.trainstep += 1