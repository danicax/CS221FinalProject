import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import talib
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import datetime


#Step 2: Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Critic 2
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.layer_1(sa))
        q1 = torch.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = torch.relu(self.layer_4(sa))
        q2 = torch.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2


from collections import namedtuple
 
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward','done'))
 
#Step 3: Define TD3 Agent


    
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def remember(self, experience):
        """Save an experience to memory."""
        self.memory.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
# Assuming you have defined your DRL agent class with methods like 'select_action', 'optimize_model', etc.
class TD3Agent:
    def __init__(self, numtickers, data, learning_rate=1e-4, tau=0.005,gamma=0.99, memory_size=10000, batch_size=128, policy_noise=0.2, noise_clip=0.5, policy_freq=4, target_update_interval=10):
        self.numtickers = numtickers
        #self.k_max = k_max
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.target_update_interval = target_update_interval
        self.state_holding=data[:,:1+2*numtickers+1]
        self.state_tensor=data[:,1+2*numtickers+1:]
        train_size = int(0.8 * len(data))
        self.train_data = self.state_tensor[:train_size]
        self.validation_data = self.state_tensor[train_size:]
        self.train_holding=self.state_holding[:train_size]
        self.validation_holding=self.state_holding[train_size:]
        self.input_size=self.state_tensor.shape[1]+numtickers+1
        self.output_size=numtickers
        self.max_action=1
        self.total_it = 0
        self.tau=tau
        
        # Assuming data contains state information and is pre-processed
        # Will add portfolio rate for each tickers into the state later. 
        self.state_dim = self.train_data.shape[1]+1+numtickers
        self.action_dim = numtickers  # One continuous action per ticker

        # Initialize policy and critic networks
        self.policy_network = Actor(self.state_dim, self.action_dim).to(device)
        self.policy_network_target = Actor(self.state_dim, self.action_dim).to(device)
        self.policy_network_target.load_state_dict(self.policy_network.state_dict())
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.actor = Actor(self.input_size, self.output_size).to(device)
        self.actor_target = Actor(self.input_size, self.output_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters())    
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = ReplayMemory(memory_size)
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Normalize training and validation data
        self.train_mean = torch.mean(self.train_data, dim=0)
        self.train_std = torch.std(self.train_data, dim=0)
        self.train_norm = (self.train_data - self.train_mean) / self.train_std
        self.validation_norm = (self.validation_data - self.train_mean) / self.train_std

    def select_action(self, normalized_action,current_holding):
        # Extract necessary components from the state
        currentbalance = current_holding[1]
        currentholding = current_holding[2:1+self.numtickers+1]
        closing_prices = current_holding[1+self.numtickers+1:2+2*self.numtickers+1]

        # Placeholder for raw action values which should come from a policy network
        # For demonstration, we randomly generate action values within the range [-K_max, K_max]
        #normalized_action = self.policy_network(currentstate.unsqueeze(0))
        #normalized_action=self.actor(currentstate).tolist()#.cpu().data.numpy().flatten()
        action = []
        # Calculate the maximum number of shares that can be bought given the balance
        for j,i in enumerate(normalized_action):
            i=round(i,2)
            if i>0:
                action.append(int((i *0.05* currentbalance)/closing_prices[j]))
            elif i<=0:
                action.append(int(i * currentholding[j]))
            else:
                action.append(0)
        
        actions = torch.tensor(action)
        #actions = torch.clamp(action_values, min=-currentholding, max=max_buy)
        # Generate actions ensuring they are within the holding limits and balance
        # Calculate total action cost
        '''
        total_action_cost = torch.dot(actions.squeeze(), closing_prices.squeeze())

        # If the total cost of buying exceeds the current balance, scale down the buy actions
        if total_action_cost > currentbalance:
            scaling_factor = currentbalance / total_action_cost
            actions = (actions.float() * scaling_factor).int()

        # Ensure actions for selling do not exceed current holdings     
        '''  
        return actions

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0).to(device)
        action_batch = torch.cat([s.unsqueeze(0) for s in batch.action], dim=0).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0).to(device)

        # Update Critic
        with torch.no_grad():
            # Adding noise to the action for the next state for target policy smoothing
            #noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
            # Ensure action_batch is of float type for noise addition
            action_batch_float = action_batch.float()
            noise = (torch.randn_like(action_batch_float) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)

            next_action = (self.policy_network_target(non_final_next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(non_final_next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch.unsqueeze(1) + self.gamma * target_Q * non_final_mask.unsqueeze(1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            
            # Compute actor loss
            q1, q2 = self.critic(state_batch, action_batch)
            actor_loss = -q1.mean()
            #actor_loss = -self.critic.q1(state_batch, self.policy_network(state_batch)).mean()
            print(f'critical_loss:{critic_loss},actor_loss:{actor_loss}')
            # Optimize the actor
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.policy_network.parameters(), self.policy_network_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        
    def remember(self, state, action, next_state, reward,done):
        """Store experience in replay memory."""
        experience = Experience(state, action,next_state, reward,  done)
         
        self.memory.remember(experience)
        #self.memory.append((state, action, reward, next_state, done))
        
    def update_state_and_calculate_reward(self,action, current_features,next_state_features,current_holding, next_state_holding):
        currentbalance=current_holding[1]
        currentholding=current_holding[2:1+self.numtickers+1]
        closing_prices=current_holding[1+self.numtickers+1:1+2*self.numtickers+1]
        new_closing_prices=next_state_holding[1+self.numtickers+1:1+2*self.numtickers+1]
        #action_tensor = torch.tensor(action, dtype=currentholding.dtype, device=currentholding.device)

        action_tensor = action
        # Step 1: Calculate new balance
        newbalance = currentbalance-np.dot(action.squeeze(), closing_prices.squeeze())
        
        # Step 2: Update holdings based on the action
        newholding = currentholding + action_tensor
        new_port = newholding*new_closing_prices
        new_total_port=new_port.sum()+newbalance
        new_total_port_ratio=torch.tensor([newbalance.tolist()]+new_port.tolist())/new_total_port
        
        # Step 3: Calculate the reward
        # Reward is the change in portfolio value based on new holdings at new closing prices
        current_total_port=(currentholding*closing_prices).sum()+currentbalance
        #reward = np.dot(newholding, new_closing_prices-closing_prices)
        reward = (new_total_port-current_total_port)/current_total_port
        #old_portfolio_value = np.dot(newholding, )
        #reward = new_portfolio_value - old_portfolio_value
        
        # Step 4: Update the state for the next timestep
        # Assuming you have a function to get the new state features for the next timestep
        
        
        # Prepare the next state with the updated balance and holdings
        #newbalance = torch.tensor([newbalance])
        next_state_features[0:1+self.numtickers]=new_total_port_ratio
        next_state_holding[1:1+self.numtickers+1]=torch.tensor([newbalance.tolist()]+newholding.tolist())
        #next_state_holding[0]=next_state_holding[0]
        return next_state_features,next_state_holding,reward
    
    def save_model(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_actor.pth")
        torch.save(self.critic.state_dict(), file_name + "_critic.pth")
    
        
    def generate_experience(self):
        #replay_memory = deque(maxlen=10000)

        def generate_episode(data_tensor, start_index, episode_length):
            end_index = min(start_index + episode_length, data_tensor.size(0))
            return data_tensor[start_index:end_index]
        
        episode_length = 150  # Length of each episode
        stride = 30  # Determines overlap between episodes
        
        def check_if_done(timestep, max_timesteps, portfolio_value, stop_loss=float('-inf'), take_profit=float('inf')):
            # Check if end of data is reached
            if timestep >= max_timesteps - 2:
                return True
            # Check for stop-loss or take-profit conditions
            if portfolio_value <= stop_loss or portfolio_value >= take_profit:
                return True

            # Other conditions can be added here
            return False

        episodes = []
        tensor_to_add = torch.tensor([1]+[0]*self.numtickers)
        
        for i in range(0, self.train_norm.size(0) - episode_length + 1, stride):         
            feature= generate_episode(self.train_norm, i, episode_length)
            tensor_to_add_repeated = tensor_to_add.repeat(feature.shape[0], 1)
            episodes.append((torch.cat((tensor_to_add_repeated,feature),dim=1),generate_episode(self.train_holding, i, episode_length)))
            #episodes['holding'].append(generate_episode(self.train_holding, i, episode_length))
        # Assuming 'state_tensor' is a tensor containing all the states for the episodes
        
        
        for j,episode in enumerate(episodes):
            features = episode[0]
            holding = episode[1]
            current_holding = holding[0]
            current_features = features[0]
            for index_val in range(len(holding)-1):               
                
                next_state_features = features[index_val+1, :]
                next_state_holding = holding[index_val+1, :]
                # Assuming a function to check if it's the end of an episode
                #done = is_end_of_episode(next_index)

                # Update state and calculate reward based on your function
                normalized_action=self.actor(current_features).tolist()
                action=agent.select_action(normalized_action,current_holding)
                next_state_features,next_state_holding,reward = agent.update_state_and_calculate_reward(action, current_features,next_state_features,current_holding, next_state_holding)
                #next_state_holding,reward 
                done = check_if_done(index_val, len(holding),reward)
                # Remember the experience
                agent.remember(current_features, torch.tensor(normalized_action), next_state_features, reward,done)
                
                current_features=next_state_features
                current_holding=next_state_holding
                self.optimize_model()
                if done:
                    break
            if j % 20 == 0:
                self.save_model(f'/{self.current_time}_model_chekpoint/{j}')
                # Move to the next state
            #if j % self.TARGET_UPDATE == 0:
            #    self.update_target_network()
    def validation(self,path):
        self.actor.load_state_dict(torch.load(f'./{path}/model_chekpoint_2023-11-17_15-14-41_80_actor.pth'))
        self.actor.eval()
        tensor_to_add = torch.tensor([1]+[0]*self.numtickers)
        tensor_to_add_repeated = tensor_to_add.repeat(self.validation_data.shape[0], 1)
        validation_data=torch.cat((tensor_to_add_repeated,self.validation_data),dim=1)
        current_features = validation_data[0, :]
        current_holding = self.validation_holding[0, :]
        reward_rec=[]

        
        for index_val in range(len(self.validation_holding)-1):               
                        
            next_state_features = validation_data[index_val+1, :]
            next_state_holding = self.validation_holding[index_val+1, :]
            # Assuming a function to check if it's the end of an episode
            #done = is_end_of_episode(next_index)

            # Update state and calculate reward based on your function
            normalized_action=self.actor(current_features).tolist()
            print(normalized_action)
            action=agent.select_action(normalized_action,current_holding)
            next_state_features,next_state_holding,reward = agent.update_state_and_calculate_reward(action, current_features,next_state_features,current_holding, next_state_holding)
            reward_rec.append(reward)
            current_features=next_state_features
            current_holding=next_state_holding            
            #next_state_holding,reward 
        return reward
            
def datapre(ticker,capital):
    # Loop through each ticker to calculate indicators
    data = yf.download(tickers)
    data = data.dropna()
    indicators={}
    for ticker in tickers:
        # Select your data for the current ticker
        close = data['Close', ticker].astype('double').values
        high = data['High', ticker].astype('double').values
        low = data['Low', ticker].astype('double').values
        volume = data['Volume', ticker].astype('double').values
        captial = 100000*np.ones(len(close))
        # Calculate each indicator for the ticker        
        indicators[ticker] = {#'close':close,
            'logreturn':np.insert(np.log(close[1:] / close[:-1]), 0, np.nan),
            'logreturn5':np.insert(np.log(close[5:] / close[:-5]), 0, [np.nan] * 5),
            'logreturn10':np.insert(np.log(close[10:] / close[:-10]), 0, [np.nan] * 10),
            'RSI7': talib.RSI(close, timeperiod=7),
            'RSI': talib.RSI(close, timeperiod=14),
            'SMA': talib.SMA(close, timeperiod=14),
            'EMA': talib.EMA(close, timeperiod=14),
            '%K': talib.STOCH(high, low, close, fastk_period=14)[0],
            'MACD': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],
            'AD': talib.AD(high, low, close, volume),
            'OBV': talib.OBV(close, volume),
            'ROC': talib.ROC(close, timeperiod=10),
            'William %R': talib.WILLR(high, low, close, timeperiod=14),
            'Disparity Index': 100 * (close - talib.EMA(close, timeperiod=14)) / talib.EMA(close, timeperiod=14)}
    close=data['Close'].astype('double').values
    ticker_matrices = [np.column_stack([v for k, v in ticker_data.items()]) for ticker_data in indicators.values()]
    data_matrix_3d = np.stack(ticker_matrices, axis=1)

    # Convert your existing 3D NumPy array to a tensor
    data_tensor_3d = torch.from_numpy(data_matrix_3d)  # Shape: (5356, 4, 11)
    dates=np.array(data.index).astype(str)
    dates_only = np.array([date.split("T")[0].replace('-','') for date in dates]).astype('int')
    timestamps_tensor = torch.tensor(dates_only).unsqueeze(1)#.unsqueeze(1)
    #capital = 1e6
    num_days = timestamps_tensor.shape[0]
    num_tickers=len(tickers)
    capital_column = np.full((num_days, 1), capital)
    holdings_columns = np.zeros((num_days, num_tickers))
    combined_array = torch.tensor(np.hstack([capital_column,holdings_columns,close]))
    capital_holdings_with_time_tensor = torch.cat([timestamps_tensor, combined_array], dim=1)  # Shape: [days, 1, features + 1]
    capital_holdings_with_time_tensor
    def preprocess_state(portfolio, features):
        # Flatten the portfolio and feature tensors
        portfolio_flat = portfolio.flatten(start_dim=1)
        features_flat = features.flatten(start_dim=1)
        # Replace nan values with an appropriate value, such as 0 or the mean of non-nan values
        #features_flat = torch.nan_to_num(features_flat, nan=0.0)
        # Concatenate the portfolio and features along the second dimension
        state = torch.cat([portfolio_flat, features_flat], dim=1)
        state = state.to(torch.float32)
        return state

    # Sample usage:
    #state_dict = {'portfolio': tensor(...), 'features': tensor(...)}
    return preprocess_state(capital_holdings_with_time_tensor, data_tensor_3d)[33:]
# Experience Replay Memory


if __name__=="__main__":
    tickers = ['QQQ', 'SPY', 'IWM', 'TLT']

    data=datapre(tickers,1000000)
    #state=data[:,1:]
    # Initialize the DRL agent
    num_tickers = len(tickers)
    agent = TD3Agent(len(tickers),data)
    agent.generate_experience();
    #agent.validation('model1')
    
        