import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # the saving directory for train.py
    parser.add_argument(
        '--output_dir', type=str, default='trained_models/GST_predictor_rand')

    # resume training from an existing checkpoint or not
    parser.add_argument(
        '--resume', default=False, action='store_true')
    # if resume = True, load from the following checkpoint
    parser.add_argument(
        '--load-path', default='trained_models/GST_predictor_non_rand/checkpoints/41200.pt',
        help='path of weights for resume training')
    parser.add_argument(
        '--overwrite',
        default=True,
        action='store_true',
        help = "whether to overwrite the output directory in training")
    parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help="number of threads used for intraop parallelism on CPU")
    # only implement in testing
    parser.add_argument(
        '--phase', type=str, default='test')

    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    # only works for gpu only (although you can make it work on cpu after some minor fixes)
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--seed', type=int, default=425, help='random seed (default: 1)')

    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training processes to use (default: 16)')

    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=2,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=30,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True,
        help='use a recurrent policy')

    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=5,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--lr', type=float, default=4e-5, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    # 10e6 for holonomic, 20e6 for unicycle
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=20e6,
        help='number of environment steps to train (default: 10e6)')
    # True for unicycle, False for holonomic
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')

    # for srnn only
    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')
    parser.add_argument(
        '--aux-loss',
        action='store_true',
        default=False,
        help='auxiliary loss on human nodes outputs')


    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=3,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=256,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Sequence length')

    # use self attn in human states or not
    parser.add_argument('--use_self_attn', type=bool, default=True,
                        help='Attention size')

    # use attn between humans and robots or not (todo: implment this in network models)
    parser.add_argument('--use_hr_attn', type=bool, default=True,
                        help='Attention size')

    # No prediction: for orca, sf, old_srnn, selfAttn_srnn_noPred ablation: 'CrowdSimVarNum-v0',
    # for constant velocity Pred, ground truth Pred: 'CrowdSimPred-v0'
    # gst pred: 'CrowdSimPredRealGST-v0'
    parser.add_argument(
        '--env-name',
        default='CrowdSimPredRealGST-v0',
        help='name of the environment')


    # sort all humans and squeeze them to the front or not
    parser.add_argument('--sort_humans', type=bool, default=True)


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args