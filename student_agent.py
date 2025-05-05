import numpy as np
import gym
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
from collections import deque
import random
import cv2
import pickle
import time

# Do not modify the input of the 'act' function and the '__init__' function. 
ORIGINAL_IMAGE_SIZE = (256, 240)
PADDED_IMAGE_SIZE = (256, 256)
INPUT_IMAGE_SIZE = (96, 96)
OUTPUT_DIM = 12 #action space

STACK_SIZE = 4
SKIP_STEP = 4

LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-6
NUM_EPISODES = 2000
EPISODE_MAX_STEPS = 2500
device = torch.device('cpu')
print(device)
class stackedReplayBuffer:
    def __init__(self, dummy_instance = None, max_size=50000):
        self.max_size = max_size
        self.dummy = dummy_instance
        self.buffer = deque(maxlen=max_size)
        self.frame_stack = deque(maxlen=STACK_SIZE)
    
    def get_len(self):
        return len(self.buffer)

    def capacity_adjust(self, Mem_size = 8): #8GB
        #to use the memory efficiently, decide capacity based on dummy instance
        cap = int(Mem_size*1024*1024*1024/np.prod(self.dummy.shape))
        self.max_size = cap
        self.buffer = deque(maxlen=cap)

    def process_image(self, input_image):
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        #padding the image to 256*240 -> 256x256
        input_image = cv2.copyMakeBorder(input_image, 8, 8, 0, 0, cv2.BORDER_CONSTANT, value=0)
        input_image = cv2.resize(input_image, INPUT_IMAGE_SIZE[:2]) #(96, 96)
        input_image = input_image.astype(np.float32) / 255.0

        return input_image
    
    def stack_update(self, input_image, is_new_episode):
        processed_image = self.process_image(input_image)
        if is_new_episode or len(self.frame_stack) == 0:
            # Clear frame stack
            self.frame_stack = deque([processed_image] * STACK_SIZE, maxlen=STACK_SIZE)
        else:
            # Append frame to stack
            self.frame_stack.append(processed_image)
        # stack to get a image(4, 96, 96)
        stacked_state = np.stack(self.frame_stack, axis=0)
        #print("stacked", stacked_state.shape)
        return stacked_state

    def add(self, experience): #(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def random_sample(self, batch_size): 
        return random.sample(self.buffer, batch_size)
    
    def increasing_distribution(self, batch_size, steepness = 0.15):
        cur_buf_size = len(self.buffer)
        #p = np.array([np.exp(steepness * i) for i in range(cur_buf_size)]) #unstable...
        p = np.linspace(cur_buf_size//2, cur_buf_size, cur_buf_size)
        p = p / np.sum(p)
        indices = np.random.choice(cur_buf_size, batch_size, p=p)
        return [self.buffer[i] for i in indices]

class QNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.input_dim = input_dim # (4 * H * W )
        self.output_dim = output_dim # 12 actions
        self.in_channel = self.input_dim[0]

        # feature extraction
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ELU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ELU(),
            torch.nn.Flatten()
        )
        # first we only trained for values of each action. TODO: dueling network
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(self._get_flattened_size(), 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 12),
        )
        # then we trained for advantage 
        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(self._get_flattened_size(), 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 1)
        )

    def _get_flattened_size(self):
        """
        Compute the size of the flattened output after convolutional layers.
        """
        # Create a dummy input tensor with shape (1, channels, height, width)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_dim)
            x = dummy_input
            # Pass through convolutional layers only
            for layer in self.feature_layer:
                x = layer(x)
            # Compute flattened size: out_channels * height * width
            return x.numel() // x.shape[0]  # Divide by batch size

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0).to(device)
        x = self.feature_layer(x)
        v = self.value_layer(x)
        adv = self.advantage_layer(x)
        q = v + adv - adv.mean(dim=1, keepdim=True)
        return q

class mario_agent():
    def __init__(self):
        self.target_net = QNet( input_dim = (STACK_SIZE, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), output_dim = 12)
        self.value_net = QNet(input_dim = (STACK_SIZE, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), output_dim = 12)
        #print(self.target_net)
        self.target_net.to(device)
        self.value_net.to(device)

        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        self.replay_buffer = stackedReplayBuffer(max_size = 50000)
        self.train_called = 0

        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999
        self.epsilon = self.epsilon_start
        self.info_stack = deque(maxlen=100)
    
    def stable_softmax(self, x, temperature = 1.0):
        return torch.nn.functional.softmax(x / temperature, dim=1)
    
    def get_action(self, stacked_state, T = 1.0, deterministic=True, value_printer = False):

        if not deterministic:
            #if np.random.rand() < self.epsilon:
            #    return np.random.randint(0, 12)
            stacked_state = torch.FloatTensor(stacked_state).to(device)
            with torch.no_grad():
                q_values = self.value_net(stacked_state)
            distribution = self.stable_softmax(q_values, temperature = T).cpu().numpy()[0]
            if value_printer:
                print("distrbution : ", distribution)
            return np.random.choice(np.arange(12), p = distribution)
        else:
            stacked_state = torch.FloatTensor(stacked_state).to(device)
            with torch.no_grad():
                q_values = self.value_net(stacked_state)
                if value_printer:
                    print("q_value : ",q_values) #debug when testing
            return torch.argmax(q_values, dim=1).item()

    def update(self, target, tau, soft = False):
        # TODO: Implement hard update or soft update
        if soft:
            for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        else:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def learn(self, batch_size, gamma = 0.9):
        # TODO: Sample a batch from the replay buffer
        self.train_called += 1
        if self.train_called % 3 != 0: # only train every 3 steps
            return
        if len(self.replay_buffer.buffer) < batch_size:
            return
        #batch_data = self.replay_buffer.sample(batch_size)
        batch_data = self.replay_buffer.increasing_distribution(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch_data)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.IntTensor(dones).unsqueeze(1).to(device)
        #DDQN
        with torch.no_grad():
            next_q_values = self.value_net(next_states)
            next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)  # shape: [batch_size, 1]

        with torch.no_grad():
            next_q_target_values = self.target_net(next_states)
            selected_q_values = next_q_target_values.gather(1, next_actions)

        q_target = rewards + gamma * selected_q_values * (1 - dones)
        q_values = self.value_net(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.train_called) % 3000 == 0:
            self.update(self.target_net, tau = 0.0)
            
    def dump_weight(self, num_episode, folder_name = 'weights'):
        save_path = folder_name+"/model_weight_{}_value.pth".format(num_episode)
        torch.save(self.value_net.state_dict(), save_path)
        save_path = folder_name+"/model_weight_{}_target.pth".format(num_episode)
        torch.save(self.target_net.state_dict(), save_path)
    
    def load_weight(self, value_path, target_path):
        if torch.cuda.is_available():
            self.value_net.load_state_dict(torch.load(value_path))
            self.target_net.load_state_dict(torch.load(target_path))
        else:
            self.value_net.load_state_dict(torch.load(value_path, map_location=torch.device('cpu')))
            self.target_net.load_state_dict(torch.load(target_path, map_location=torch.device('cpu')))
    
    def eval(self):
        self.value_net.eval()
        self.target_net.eval()
    
    def train(self):
        self.value_net.train()
        self.target_net.train()

    def epsilon_update(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.agent = mario_agent()
        self.value_path = 'model_weight_12000_value.pth'
        self.target_path = 'model_weight_12000_target.pth'
        self.agent.load_weight(self.value_path, self.target_path)
        self.skip_cnt = 4
        self.step = 0
        self.action = 0
        self.reset_cnt = 0

        self.test_rand = 0

    def act(self, observation):
        if self.reset_cnt > 0:
            self.reset_cnt-=1
            return 0
        if self.test_rand > 0:
            self.test_rand -= 1
            return self.action_space.sample()
        if self.step % self.skip_cnt == 0:
            state = self.agent.replay_buffer.stack_update(observation, is_new_episode = False)
            self.action = self.agent.get_action(state, deterministic = True)
        self.step += 1
        return self.action
