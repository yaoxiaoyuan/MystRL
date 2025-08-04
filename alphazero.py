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

mp.set_start_method('spawn', force=True)

def self_play(task_id, env, model, args):
    """
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
            z = (play_data[i][j][1] * mcts.root_list[i].winner)
            z = (player * mcts.root_list[i].winner)
            play_data_with_result.append((state, player, pp, z))

    logger.info(f"task {task_id} self play done.") 
    return play_data_with_result
 

def train_az(env_cls, args):
    """
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


def test_play(env_cls, args):
    """
    """
    env = env_cls(args)

    model = ResNet(
                 input_shape=env.input_shape,
                 n_filters=args.n_filters,
                 n_res_blocks=args.n_res_blocks,
                 hidden_size=args.hidden_size,
                 n_class=env.n_actions
            ).to(args.device)    
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    human = int(input("Play First Or Second(1: First, -1:Second):"))
    state = env.reset()
    player = 1
    winner = 0
    done = False 
    mcts = MCTS(env, 1, args.c_puct, model, args.n_simulations, args.device)
    while True:
        env.render(state, player, winner, done)
        mcts.simulate()
        
        #print(f"state:\n{mcts.root_list[0].state}")
        print(f"prior:{mcts.root_list[0].prior}")        
        print(f"valid_moves:{mcts.root_list[0].valid_moves}")
        print(f"values:{mcts.root_list[0].children_value}")
        print(f"visits:{mcts.root_list[0].children_visits}")

        pi = np.zeros_like(mcts.root_list[0].prior)
        pi[mcts.root_list[0].valid_moves] = mcts.root_list[0].children_visits / mcts.root_list[0].children_visits.sum()
        idx = mcts.root_list[0].children_visits.argmax()
        action = mcts.root_list[0].valid_moves[idx]
        print(f"search policy:{pi}, action:{action}")
 
        if player == human:
            action = int(input("Input action:"))
            idx = np.where(mcts.root_list[0].valid_moves==action)[0][0]
 
        mcts.root_list[0] = mcts.root_list[0].children[idx]
        parent = mcts.root_list[0].parent
        mcts.root_list[0].parent = None
        del parent
    
        state,winner,done = env.step(state, action, player) 
        player = -player
        if done:
            env.render(state, player, winner, done)
            human = int(input("Play First Or Second(1: First, -1:Second):"))
            state = env.reset()                                             
            player = 1                                                      
            winner = 0                                                      
            done = False
            mcts = MCTS(env, 1, args.c_puct, model, args.n_simulations, args.device)

def build_argparser():
    """
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
                        default=1000,
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
    """
    parser = build_argparser()

    if add_custom_argument_func:
        parser = add_custom_argument_func(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.mode == "train":
        train_az(env_cls, args)
    elif args.mode == "test":
        test_play(env_cls, args)
