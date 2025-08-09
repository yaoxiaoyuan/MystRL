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
import sys
import os
import math
import logging
import random
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from mcts import MCTS
from resnet import ResNet
from logger import logger, add_file_handlers, print_formated_args
from board_game import GameState

mp.set_start_method('spawn', force=True)

def self_play(task_id, env, model, args):
    """
    Executes self-play games using Monte Carlo Tree Search (MCTS) for policy improvement.
    Generates training data (state, player, policy, value) through parallel game simulations.
    
    Parameters:
    task_id (int)     : Identifier for logging and tracking individual tasks
    env (object)      : Game environment defining rules and valid moves
    model (nn.Module) : Neural network for predicting policies and values
    args (object)     : Configuration parameters container
    
    Returns:
    list : Tuples of (game_state, active_player, policy_vector, outcome)
           - outcome: +1 for win, -1 for loss from player's perspective
    """ 
    play_data = [
            []
            for _ in range(args.n_parallel_games)
        ]
 
    mcts = MCTS(env, args.n_parallel_games, args.c_puct, model, args.n_simulations, args.device) 
 
    steps = 1                                                                                        
    while not mcts.is_done():                                                                        
                                                                                                     
        mcts.simulate()                                                                              
        logger.info(f"task {task_id} {steps} simulation done.")
  
        greedy = True                                                                                             
        temperature = 0                                                                              
        add_dirichlet_noise = False                                                                  
        alpha = None                                                                                 
        eps = None                                                                                   
        if steps == 0:                                                                               
            add_dirichlet_noise = True                                                               
            alpha = args.alpha                                                                       
            eps = args.eps                                                                           
        if steps < args.initial_exploration_moves:                                                   
            temperature = args.temperature                                                           
            greedy = False
 
        for i in range(args.n_parallel_games):
            if mcts.root_list[i].is_terminal:
                continue

            if greedy:
                idx = mcts.root_list[i].children_visits.argmax()
            else:
                psa = np.power(mcts.root_list[i].children_visits, 1/temperature)
                psa = psa / psa.sum()
                if add_dirichlet_noise:
                    noise = np.random.dirichlet([alpha]*len(mcts.root_list[i].valid_moves)) 
                    psa = eps * psa + (1 - eps) * noise
                idx = np.random.choice(list(range(len(mcts.root_list[i].valid_moves))), size=1, p=psa)[0]

            pi = np.zeros_like(mcts.root_list[i].prior)
            pi[mcts.root_list[i].valid_moves] = mcts.root_list[i].children_visits / mcts.root_list[i].children_visits.sum()
            play_data[i].append((mcts.root_list[i].state, mcts.root_list[i].player, pi))

            mcts.root_list[i] = mcts.root_list[i].children[idx]
            parent = mcts.root_list[i].parent
            mcts.root_list[i].parent = None
            del parent

        logger.info(f"task {task_id} {steps} move done.") 
        steps += 1
    
    play_data_with_result = []
    for i in range(args.n_parallel_games):
        for j in range(len(play_data[i])):
            state, player, pp = play_data[i][j]
            z = (player * mcts.root_list[i].winner)
            play_data_with_result.append((state, player, pp, z))

    logger.info(f"task {task_id} self play done.") 
    return play_data_with_result
 

def train_az(env_cls, args):
    """
    Train an AlphaZero-style model using self-play data and iterative updates.

    Parameters:
    env_cls (callable): Environment class constructor. Must accept `args` parameter
        and provide properties: `input_shape`, `n_actions`, and method `convert_to_model_inputs`
    args (argparse.Namespace): Configuration object containing:
        - device (torch.device): Computation device (CPU/GPU)
        - task_name (str): Identifier for logging
        - n_filters (int): Convolutional filter count
        - n_res_blocks (int): Number of residual blocks
        - hidden_size (int): Neural network hidden layer size
        - n_total_games (int): Total games to generate
        - n_parallel_games (int): Games per parallel process
        - n_processes (int): Number of worker processes
        - train_epochs (int): Training epochs per iteration
        - batch_size (int): Minibatch size for SGD
        - lr (float): Adam optimizer learning rate
        - save_path (str): Directory for model checkpoints
    """
    logger_path = os.path.join("logger", f"{args.task_name}.log")
    add_file_handlers(logger_path)
    print_formated_args(args)

    env = env_cls(args)
    model = ResNet(
                 input_shape=env.input_shape,
                 n_filters=args.n_filters,
                 n_res_blocks=args.n_res_blocks,
                 hidden_size=args.hidden_size,
                 n_class=env.n_actions
            ).to(args.device)
 
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Model Params:{total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for i in range(args.n_total_games//(args.n_parallel_games*args.n_processes)):

        logger.info(f"{i}-th self play start.") 

        tasks = [(task_id, env, model, args) for task_id in range(args.n_processes)]
        with mp.Pool(processes=args.n_processes) as pool:
            result_lists = pool.starmap(self_play, tasks)
 
        play_data = [item for sublist in result_lists for item in sublist]

        logger.info(f"{i}-th self play done, total data:{len(play_data)}.") 
       
        model.train() 
        for epoch in range(args.train_epochs):
            random.shuffle(play_data)
            for step in range(len(play_data) // args.batch_size):
                start = step * args.batch_size
                end = start + args.batch_size
                x = [] 
                pi = []
                z = [] 
                for state,player,pp,zz in play_data[start:end]:
                    x.append(env.convert_to_model_inputs(state, player))
                    pi.append(pp)
                    z.append(zz)
                x = np.stack(x, axis=0)
                pi = np.stack(pi, axis=0)
                z = np.stack(z, axis=0)

                x = torch.from_numpy(x).float().to(args.device)
                pi = torch.from_numpy(pi).float().to(args.device)
                z = torch.from_numpy(z).float().to(args.device)

                logits, value = model(x)
                loss_1 = F.mse_loss(value, z) 
                loss_2 = F.cross_entropy(logits, pi)

                logger.info(f"iteration {i} epoch {epoch} step {step}, value loss:{loss_1.item()}")
                logger.info(f"iteration {i} epoch {epoch} step {step}, policy loss:{loss_2.item()}")

                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        logger.info("save model now.")
        save_dir = os.path.join(args.save_path, f"iteration-{i}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_weights")
        torch.save(model.state_dict(), save_path)
        logger.info("save model done.")

def print_node_stats(node):
    """
    Prints statistical information of a Monte Carlo Tree Search (MCTS) node.
    
    Displays key node attributes including:
    - Prior probability from the neural network policy
    - Valid legal moves available at this game state
    - Value estimates (Q-values) for child nodes
    - Visit counts for child nodes during simulations
    
    Args:
        node (MCTSNode): Node object from the MCTS tree structure
    """
    print(f"prior:{node.prior}")
    print(f"valid_moves:{node.valid_moves}")
    print(f"values:{node.children_value}")
    print(f"visits:{node.children_visits}")

def test_play(env_cls, args):
    """
    Executes interactive gameplay test loop with human-AI interaction.
    
    Initializes game environment and loads pre-trained ResNet model for MCTS.
    Supports two modes:
    - GUI mode with PyGame rendering
    - Console mode without visualization
    
    Args:
        env_cls (class): Environment class constructor
        args (Namespace): Configuration parameters containing:
            render_mode (str): "gui" or "console" display mode
            device (torch.device): Computation device (CPU/GPU)
            model_path (str): Path to trained model weights
            n_simulations (int): MCTS simulation count per move
            c_puct (float): Exploration constant for UCT formula
    """
    env = env_cls(args)
    
    if args.render_mode == "gui":
        env.init_pygame()
    
    model = ResNet(
                 input_shape=env.input_shape,
                 n_filters=args.n_filters,
                 n_res_blocks=args.n_res_blocks,
                 hidden_size=args.hidden_size,
                 n_class=env.n_actions
            ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    running = True
    game_state = GameState(env.reset(), args.render_mode)
    while running:

        env.render(game_state)
        running, reset, chosen_turn, click_pos = env.process_input(game_state)
        
        if reset:
            game_state.reset(env.reset(), chosen_turn)
            mcts = MCTS(env, 1, args.c_puct, model, args.n_simulations, args.device)

        elif not game_state.done:

            action = None
            if game_state.player == game_state.human:
                if click_pos is not None:
                    possible_action = env.get_possible_action(game_state, click_pos)
                    if possible_action is not None: 
                        idx = np.where(mcts.root_list[0].valid_moves==possible_action)[0]
                        if idx.size != 0:
                            action = possible_action
            else: 
                if game_state.turn == 0:
                    game_state.thinking = True
                    env.render(game_state)
                    mcts.simulate()
                    game_state.thinking = False
                    env.render(game_state)

                print(f"----------turn:{game_state.turn}----------")
                print_node_stats(mcts.root_list[0])

                pi = np.zeros_like(mcts.root_list[0].prior)
                pi[mcts.root_list[0].valid_moves] = mcts.root_list[0].children_visits / mcts.root_list[0].children_visits.sum()
                print(f"search policy:{pi}")
                idx = mcts.root_list[0].children_visits.argmax()
                action = mcts.root_list[0].valid_moves[idx]

            if action is not None:
                 
                game_state.state, game_state.winner, game_state.done = env.step(
                    game_state.state, action, game_state.player)
                game_state.player = -game_state.player
                game_state.turn += 1
                env.update_last_move(game_state, action)

                env.render(game_state)
                
                if game_state.turn == 1 and game_state.player != game_state.human:
                    game_state.thinking = True
                    env.render(game_state)
                    mcts.simulate()
  
                idx = np.where(mcts.root_list[0].valid_moves==action)[0][0]
                mcts.root_list[0] = mcts.root_list[0].children[idx]
                parent = mcts.root_list[0].parent
                mcts.root_list[0].parent = None
                del parent
 
                if game_state.player != game_state.human:
                    game_state.thinking = True
                    env.render(game_state)
                    mcts.simulate()
                    game_state.thinking = False
                    env.render(game_state)

def build_argparser():
    """
    Constructs and configures a command-line argument parser for the RL application.
    
    Defines parameters for environment configuration, training/inference settings, 
    model architecture, and algorithm hyperparameters. Uses argparse.RawTextHelpFormatter 
    to preserve formatting in help messages.

    Returns:
        argparse.ArgumentParser: Configured parser object with all supported arguments.
    """
    description=(
            "mystRL is a simple Python implementation of RL."
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
                        default="run",
                        help="Name of the environment/task to run")

    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "test"],
                        default="train",
                        help="Operating mode: 'train' for training, 'test' for evaluation")

    parser.add_argument("--render_mode",
                        choices=["gui", "text"],
                        default="gui",
                        help="if gui, use pygame to show")

    parser.add_argument("--model_path",
                        type=str,
                        default=None,
                        help="Path to load pre-trained model weights")
    
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="")

    parser.add_argument("--batch_size",                                                                           
                        type=int,                                                                       
                        default=512,                                                                     
                        help="")

    parser.add_argument("--train_epochs",
                        type=int,  
                        default=1, 
                        help="")

    parser.add_argument("--n_total_games",
                        type=int,
                        default=1000000,
                        help="")

    parser.add_argument("--n_parallel_games",
                        type=int,
                        default=125,
                        help="")

    parser.add_argument("--n_processes",
                        type=int,
                        default=8,
                        help="")

    parser.add_argument("--c_puct",
                        type=float,
                        default=2,
                        help="")
 
    parser.add_argument("--n_simulations",
                        type=int,
                        default=500,
                        help="")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="Compute device: 'cpu' or 'cuda'")

    parser.add_argument("--save_path",
                        type=str,
                        default="model",
                        help="Directory for saving model checkpoints")

    parser.add_argument("--n_filters",
                        type=int,
                        default=256,
                        help="")

    parser.add_argument("--n_res_blocks",
                        type=int,
                        default=6,
                        help="")

    parser.add_argument("--hidden_size",
                        type=int,
                        default=256,
                        help="")                 

    parser.add_argument("--initial_exploration_moves",
                        type=int,
                        default=16,
                        help="")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.03,
                        help="")

    parser.add_argument("--eps",
                        type=float,
                        default=0.25,
                        help="")

    parser.add_argument("--temperature",
                        type=float,
                        default=1,
                        help="")

    return parser


def main(env_cls, add_custom_argument_func=None):
    """
    Entry point for executing RL training or evaluation workflows.
    
    Parses command-line arguments, optionally extends them through a custom function, 
    and dispatches to the appropriate workflow handler based on the selected mode.

    Args:
        env_cls (class): Environment class to be used for RL tasks
        add_custom_argument_func (function, optional): Callback for adding custom arguments 
            to the parser. Expected signature: func(parser) -> modified_parser.
    """
    parser = build_argparser()

    if add_custom_argument_func:
        parser = add_custom_argument_func(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.mode == "train":
        train_az(env_cls, args)
    elif args.mode == "test":
        test_play(env_cls, args)
