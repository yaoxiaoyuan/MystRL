#encoding=utf-8
#◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Author: Xiaoyuan Yao
# GitHub: https://github.com/yaoxiaoyuan/mystRL/
# Contact: yaoxiaoyuan1990@gmail.com
# Created: Sat Jun 14 15:08:00 2025
# License: MIT
# Version: 0.1.0
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import os
import sys
import math
import random
import time
from collections import deque
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from logger import logger, add_file_handlers, print_formated_args
                                                                                                          
class ReplayBuffer:
    def __init__(self, args, state_info):
        """
        Initialize experience replay buffer with support for multiple state arrays
        
        Parameters:
            buffer_size (int): Maximum capacity of the buffer
            state_info (list): List of dicts specifying state array shapes and dtypes
            batch_size (int): Size of batch to sample
        """
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.n_step = args.n_step
        self.gamma = args.gamma

        self.count = 0  # Current number of stored experiences
        self.pos = 0
        self.discount = np.power(self.gamma, np.arange(self.n_step))

        # Temporary buffers for n-step accumulation
        self.state_buffers = [
            deque(maxlen=self.n_step) for _ in range(len(state_info))
        ]
        self.next_state_buffers = [
            deque(maxlen=self.n_step) for _ in range(len(state_info))
        ]
        self.action_buffer = deque(maxlen=self.n_step)
        self.reward_buffer = deque(maxlen=self.n_step)
        self.done_buffer = deque(maxlen=self.n_step)
        
        # Pre-allocate buffer arrays for each state type
        self.states = []
        self.next_states = []
        
        # Create storage arrays for each state type
        for info in state_info:
            dtype = np.int32 if info['dtype'] == 'int' else np.float32
            self.states.append(np.zeros(
                (self.buffer_size, *info['shape']), dtype=dtype
            ))
            self.next_states.append(np.zeros(
                (self.buffer_size, *info['shape']), dtype=dtype
            ))
        
        # Regular buffers for other experience components
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        
    def store(self, state, action, reward, next_state, done):
        """
        Store experience in the buffer
        
        Parameters:
            state (list): List of current state arrays
            action: Action taken
            reward: Reward received
            next_state (list): List of next state arrays
            done: Termination flag (0/1)
        """
        # Add to temporary buffers
        for i in range(len(state)):
            self.state_buffers[i].append(state[i])
            self.next_state_buffers[i].append(next_state[i])
            
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

        # Save experience when we have a full n-step or episode terminates
        if len(self.state_buffers[0]) >= self.n_step or done:
            # Calculate n-step return
            rewards = np.array([r for r in self.reward_buffer])
            n_step_return = (rewards * self.discount[:len(rewards)]).sum()
            
            # Save initial state arrays
            for i in range(len(self.states)):
                self.states[i][self.pos] = self.state_buffers[i][0]
            
            # Save final next_state (current terminal state)
            for i in range(len(self.next_states)):
                self.next_states[i][self.pos] = next_state[i]
                
            self.actions[self.pos] = self.action_buffer[0]
            self.rewards[self.pos] = n_step_return
            self.dones[self.pos] = done
            
            # Update buffer position and count
            self.count = min(self.count + 1, self.buffer_size) 
            self.pos = (self.pos + 1) % self.buffer_size

        # Maintain buffer sizes after storing
        if len(self.state_buffers[0]) >= self.n_step:
            self.action_buffer.popleft()
            self.reward_buffer.popleft()
            self.done_buffer.popleft()
            for i in range(len(self.state_buffers)):
                self.state_buffers[i].popleft()
                self.next_state_buffers[i].popleft()
 
        # Reset at episode termination
        if done:
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.done_buffer.clear()
            for i in range(len(self.state_buffers)):
                self.state_buffers[i].clear()
                self.next_state_buffers[i].clear()

    def sample(self):
        """
        Sample a random batch of experiences from the buffer
        
        Returns:
            states (list), actions, rewards, next_states (list), dones
        """
        if self.count < self.batch_size:
            raise ValueError("Not enough experiences in buffer to sample")
        
        indices = random.sample(range(self.count), self.batch_size)
        
        batch_states = [arr[indices] for arr in self.states]
        batch_next_states = [arr[indices] for arr in self.next_states]
        
        return (batch_states,
                self.actions[indices],
                self.rewards[indices],
                batch_next_states,
                self.dones[indices])
 
    def __len__(self):
        return self.count


def calculate_dqn_loss(model, target_model, batch, args):
    """
    Compute DQN loss for a batch of transitions. Supports standard and Double DQN,
    with configurable loss functions (Huber or MSE).

    Args:
        model (nn.Module): Online Q-network (policy network)
        target_model (nn.Module): Target Q-network (fixed for stability)
        batch (tuple): Transition tuple containing:
            states (Tensor): Current state tensors
            actions (Tensor): Actions taken in each state
            rewards (Tensor): Immediate rewards received
            next_states (Tensor): Subsequent states after actions
            dones (Tensor): Binary terminal state indicators
        args (object): Configuration parameters with attributes:
            use_double_dqn (bool): Toggle Double DQN calculation
            gamma (float): Discount factor for future rewards
            n_step (int): Number of steps for multi-step learning
            loss (str): Loss function type ('huber' or 'mse')

    Returns:
        Tensor: Computed loss value (scalar)
    """
    states, actions, rewards, next_states, dones = batch

    # Get Q values for chosen actions (using policy network)
    q_values = model(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
     
    # Get next Q values from target network (detached)  
    with torch.no_grad(): 
        if not args.use_double_dqn: 
            next_q_values = target_model(next_states).max(1).values.detach()
        else:
            next_actions = model(next_states).argmax(1)
            next_q_values = target_model(next_states).detach().gather(1, next_actions.unsqueeze(1)).squeeze(1)

    # Calculate expected Q values using Bellman equation
    expected_state_action_values = rewards + ((args.gamma ** args.n_step) * next_q_values * (1 - dones))
    
    # Compute loss
    if args.loss == "huber":
        loss = F.huber_loss(state_action_values, expected_state_action_values)
    elif args.loss == "mse":
        loss = F.mse_loss(state_action_values, expected_state_action_values)   
   
    #print(loss)
 
    return loss
 

def get_q_values(model, state):
    """
    Calculate Q-values for a given state using the policy model.
    
    Sets the model to evaluation mode and disables gradient calculation for inference.
    Forward-passes the state through the model to compute the Q-value estimates.
    
    Parameters:
    model (torch.nn.Module): The neural network model for Q-value estimation
    state (torch.Tensor): Current environment state tensor
    
    Returns:
    torch.Tensor: Tensor containing estimated Q-values for each possible action
    """
    model.eval()
    with torch.no_grad():
        q_values = model(state)
    return q_values

def select_action(q_values, n_actions, eps):
    """  
    Select an action using an epsilon-greedy exploration strategy.
    
    With probability `eps`, chooses a random action (exploration). Otherwise selects
    the action with the highest Q-value (exploitation). Handles discrete action spaces.
    
    Parameters:
    q_values (torch.Tensor): Tensor of Q-values for the current state
    n_actions (int): Number of possible actions in the environment
    eps (float): Current epsilon value (exploration probability between 0-1)
    
    Returns:
    int: Index of the selected action (0 to n_actions-1)
    """
    # Random action with probability epsilon
    if random.random() < eps:
        return random.randint(0, n_actions-1)
        
    # Exploitation: choose best action from policy network
    else:
        return q_values.argmax(-1).item()
    

def update_epsilon(steps, args):
    """
    Update the exploration rate (epsilon) according to a decay schedule.
    
    Supports exponential decay (approaches min_eps asymptotically) or linear decay
    (decreases by fixed rate per step). Ensures epsilon doesn't drop below min_eps.
    
    Parameters:
    steps (int): Current number of training steps completed
    args (argparse.Namespace): Configuration object containing:
        - decay_type (str): 'exp' for exponential, otherwise linear
        - min_eps (float): Minimum exploration probability
        - max_eps (float): Starting exploration probability
        - decay_rate (float): Controls decay speed 
    
    Returns:
    float: Updated epsilon value
    """
    if args.decay_type == "exp":
        return max(args.min_eps, args.min_eps + (args.max_eps - args.min_eps) * np.exp(-steps / args.decay_rate))        
    else:
        return max(args.min_eps, args.max_eps - steps * args.decay_rate)


def update_lr(optimizer, episode, args):
    """
    Update learning rate for optimizer according to a schedule.
    
    Supports constant (no change), cosine annealing, or linear decay schedules.
    Updates the learning rate in-place for all parameter groups in the optimizer.
    
    Parameters:
    optimizer (torch.optim.Optimizer): Training optimizer
    episode (int): Current episode number
    args (argparse.Namespace): Configuration object containing:
        - lr_scheduler (str): 'constant', 'cosine', or 'linear'
        - total_decay_episodes (int): Episodes over which to decay LR
        - lr (float): Initial learning rate
        - min_lr (float): Minimum learning rate after decay
    """
    if args.lr_scheduler == "constant":
        return

    lr = args.min_lr
    if episode < args.total_decay_episodes:
        if args.lr_scheduler == "cosine":
            cos = math.cos(math.pi * episode / args.total_decay_episodes)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + cos)    
        elif args.lr_scheduler == "linear":
            ratio = episode / args.total_decay_episodes
            lr = args.lr - (args.lr - args.min_lr) * ratio
                
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def convert_states_to_tensors(state_list, device):
    """
    Converts a list of state arrays from ReplayBuffer samples to PyTorch tensors
    with automatic type selection (long for integer types, float for floating types)
    
    Parameters:
        state_list (list): List of numpy arrays representing states
                           as returned by ReplayBuffer.sample()
    
    Returns:
        list: List of torch.Tensor objects with appropriate data types
    """
    tensor_list = []
    
    # Process each state array in the input list
    for array in state_list:
        # Automatically determine dtype: convert integer arrays to long tensors
        if np.issubdtype(array.dtype, np.integer):
            # Use torch.long for integer data (e.g., discrete actions, IDs)
            tensor = torch.from_numpy(array).long().to(device)
        else:
            # Use torch.float for floating-point data (e.g., continuous states)
            tensor = torch.from_numpy(array).float().to(device)
        
        tensor_list.append(tensor)
    
    return tensor_list


def train_dqn(env_cls, model_cls, args):
    """
    """
    logger_path = os.path.join("logger", f"{args.task_name}.log")
    add_file_handlers(logger_path)
    print_formated_args(args)

    env = env_cls(args)
    model = model_cls(args).to(args.device)
    target_model = model_cls(args).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Model Params:{total_params}")

    target_model.load_state_dict(model.state_dict())    
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(args, env.state_info)

    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    eps = args.max_eps
    updates = 0
    episode_cnt = 0
    for episode in range(1, args.n_episodes+1):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            state_tensors = convert_states_to_tensors(state, args.device) 
            state_tensors = [t.unsqueeze(0) for t in state_tensors]
            q_values = get_q_values(model, state_tensors)
            action = select_action(q_values, env.n_actions, eps)
            
            next_state, reward, done = env.step(action)
            
            replay_buffer.store(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if len(replay_buffer) < args.min_train_buffer_size:
                continue
            eps = update_epsilon(total_steps, args)
            total_steps += 1 
            if total_steps % args.update_frequency == 0:
                for _ in range(args.n_updates):
                    states,actions,rewards,next_states,dones = replay_buffer.sample()
                    batch = (
                        convert_states_to_tensors(states, device=args.device),
                        torch.tensor(actions, dtype=torch.long, device=args.device),
                        torch.tensor(rewards, dtype=torch.float, device=args.device),
                        convert_states_to_tensors(next_states, device=args.device),
                        torch.tensor(dones, dtype=torch.float, device=args.device)
                    )

                    optimizer.zero_grad()
                    loss = calculate_dqn_loss(model, target_model, batch, args)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                    updates += 1
 
            if total_steps % args.target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict()) 

        if len(replay_buffer) >= args.min_train_buffer_size:
            episode_cnt += 1
            update_lr(optimizer, episode_cnt, args)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        logger.info(f"episode {episode}, len:{episode_length}, reward: {episode_reward}, eps: {eps}, updates: {updates}, buffer:{len(replay_buffer)}")
        
        if episode % args.save_every_episodes == 0:
            avg_reward = sum(episode_rewards) / episode
            avg_length = sum(episode_lengths) / episode
            max_reward = max(episode_rewards)
            logger.info("save model now.")
            logger.info(f"avg_reward: {avg_reward}, avg_length: {avg_length}, max_reward: {max_reward}")  
            save_dir = os.path.join(args.save_path, f"episode-{episode}")
            if not os.path.exists(save_dir):                                                                
                os.makedirs(save_dir, exist_ok=True)  
            save_path = os.path.join(save_dir, "model_weights")
            torch.save(model.state_dict(), save_path)
            logger.info("save model done.")


def test_play(env_cls, model_cls, args):
    """
    Trains a Deep Q-Network (DQN) agent on a specified environment.
    
    Parameters:
    env_cls (class): Environment class to instantiate training environment
    model_cls (class): DQN model class for Q-function approximation
    args (object/Namespace): Configuration parameters with attributes:
        - task_name (str): Task identifier for logging/saving
        - device (torch.device): Computation device (e.g., 'cuda', 'cpu')
        - n_episodes (int): Total training episodes
        - lr (float): Learning rate for optimizer
        - max_eps (float): Starting epsilon for ε-greedy policy
        - min_train_buffer_size (int): Minimum replay buffer size before training
        - update_frequency (int): Steps between network update batches
        - n_updates (int): Gradient steps per update interval
        - target_update_frequency (int): Steps between target network syncs
        - grad_clip (float): Gradient clipping threshold
        - save_every_episodes (int): Episode interval for model saving

    """
    env = env_cls(args)
    model = None
    if not args.human:
        model = model_cls(args).to(args.device)
        model.load_state_dict(torch.load(args.model_path, map_location="cpu")) 
    
    running = True
    state = env.reset()
    done = False
    env.render()
    frame_count = 0
    action = None
    while True:
        frame_count += 1
        if args.human:
            running, detect_action = env.process_input()
            if detect_action is not None:
                action = detect_action
        else:
            if frame_count == 1:
                state_tensors = convert_states_to_tensors(state, args.device)
                state_tensors = [t.unsqueeze(0) for t in state_tensors]
                q_values = get_q_values(model, state_tensors)
                action = select_action(q_values, env.n_actions, args.min_eps)
                action_name = env.act2str[action]
                q_values = q_values.flatten()                                                                     
                q_values_dic = {env.act2str[a]:q_values[a].item() for a in env.act2str}                           
                logger.info(f"q_values: {q_values_dic}, action: {action_name}, score: {env.score}") 
            if env.render_mode == "gui":
                running, _ = env.process_input()
        
        if not running:
            return
        
        if env.render_mode == "text" or frame_count >= args.fps // min(30, args.speed):
            frame_count = 0            
            state,reward,done = env.step(action)
            action = None
            
        env.render()
        
        if done:
            time.sleep(1)
            state = env.reset()
            done = False
            env.render()
            frame_count = 0
        
        if env.render_mode == "gui":
            env.clock.tick(env.fps)


def build_argparser():
    """
    Constructs and configures an argument parser for reinforcement learning tasks.
    
    Returns:
        argparse.ArgumentParser: Parser object with predefined command-line arguments for RL training/evaluation.
    """    
    description=( 
            "MystRL is a simple Python implementation of RL."
    )
    usage = (
    "For training and inference, please refer to the respective shell scripts."
    )
    parser = argparse.ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--task_name",
                        type=str,
                        default="run_rl",
                        help="Name of the task to run")
    
    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "test"],
                        default="train",
                        help="Operating mode: 'train' for training, 'test' for evaluation")
    
    parser.add_argument("--model_path",
                        type=str,
                        default=None,
                        help="Path to load pre-trained model weights")
    
    parser.add_argument("--buffer_size",
                        type=int,
                        default=200000,
                        help="Maximum capacity of the experience replay buffer")
    
    parser.add_argument("--min_train_buffer_size",
                        type=int,
                        default=10000,
                        help="Minimum experiences required before starting training")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Number of experiences sampled per training iteration")
    
    parser.add_argument("--n_episodes",
                        type=int,
                        default=100000,
                        help="Total training episodes to execute")
    
    parser.add_argument("--update_frequency",
                        type=int,
                        default=2,
                        help="Perform network updates every N environment steps")
    
    parser.add_argument("--n_updates",
                        type=int,
                        default=2,
                        help="Consecutive gradient updates per training step")
    
    parser.add_argument("--max_eps",
                        type=float,
                        default=0.999,
                        help="Maximum exploration rate (ε) starting value")
    
    parser.add_argument("--min_eps",
                        type=float,
                        default=0.01,
                        help="Minimum exploration rate (ε) lower bound")
    
    parser.add_argument("--decay_rate",
                        type=float,
                        default=20000,
                        help="Controls ε decay speed")
   
    parser.add_argument("--decay_type",
                        type=str,                                                                       
                        choices=["exp", "linear"],                                                       
                        default="exp")
       
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help="Discount factor for future rewards")
    
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate for optimizer")
   
 
    parser.add_argument("--min_lr",
                        type=float,
                        default=1e-5,
                        help="Min learning rate for optimizer")


    parser.add_argument("--lr_scheduler",
                        type=str,
                        choices=["constant", "cosine", "linear"],
                        default="constant",
                        help="learning rate scheduler")

    parser.add_argument("--grad_clip",
                        type=float,
                        default=1,
                        help="Gradient clipping threshold")
    
    parser.add_argument("--target_update_frequency",
                        type=int,
                        default=2000,
                        help="Steps between syncing main and target networks")

    parser.add_argument("--save_every_episodes",
                        type=int,
                        default=500,
                        help="Episode interval for checkpoint saves")
    
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="Compute device: 'cpu' or 'cuda'")
    
    parser.add_argument("--use_double_dqn",
                        default=False,
                        action='store_true',
                        help="Enable Double DQN (reduces overestimation bias)")
    
    parser.add_argument("--loss",
                        type=str,
                        choices=["huber", "mse"],
                        default="mse",
                        help="Loss function: 'huber' or 'mse'")
    
    parser.add_argument("--n_step",
                        type=int,
                        default=1,
                        help="Multi-step reward accumulation horizon")
    
    parser.add_argument("--save_path",
                        type=str,
                        default="model",
                        help="Directory for saving model checkpoints")
    
    parser.add_argument("--total_decay_episodes",
                        type=int,
                        default=50000,
                        help="total decay episodes from max to min")

    parser.add_argument("--n_last_frames",
                        type=int,
                        default=4,
                        help="use last n frames as input")

    parser.add_argument("--render_mode",
                        choices=["gui", "text"],
                        default="text",
                        help="if gui, use pygame to show")

    parser.add_argument("--fps",
                        type=int,
                        default=60,
                        help="fps when render_mode=gui ")

    parser.add_argument("--speed",
                        type=int,
                        default=8,
                        help="frame speed")
    
    parser.add_argument("--human",
                        default=False,
                        action='store_true',
                        help="Human plays the game when enabled; otherwise uses AI")

    return parser

def main(env_cls, model_cls, add_custom_argument_func=None):
    """
    Main entry point for DQN training or testing pipeline.
    
    Parses command-line arguments, initializes environment and model, 
    and executes either training or testing based on the provided mode.
    
    Args:
        env_cls (type): Environment class to instantiate 
        model_cls (type): DQN model class to instantiate
        add_custom_argument_func (callable, optional): Function that adds custom arguments
            to the parser. Should take an ArgumentParser and return modified parser.
    """
    parser = build_argparser()

    if add_custom_argument_func:
        parser = add_custom_argument_func(parser)
    
    args = parser.parse_args(sys.argv[1:])

    if args.mode == "train":
        train_dqn(env_cls, model_cls, args)
    elif args.mode == "test":
        test_play(env_cls, model_cls, args)        

