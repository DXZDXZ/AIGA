import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ga_iter_steps', default=150)
    # parser.add_argument('--max_reduce_steps', default=50)
    parser.add_argument('--n_select_chroms', default=2)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--n_epochs', default=1000)
    parser.add_argument('--n_anchors', default=2)
    parser.add_argument('--node_dim', default=20)
    parser.add_argument('--LR', default=3e-4)
    parser.add_argument('--eps_clip', default=0.2)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--pop_size', default=50, help="种群大小")
    parser.add_argument('--n_rollout', default=10, help="")
    parser.add_argument('--roll_out_steps', default=20, help="")
    parser.add_argument('--train_steps', default=3, help="")
    parser.add_argument('--device', default="cuda:1")
    parser.add_argument('--input_node_dim', type=int, default=5)
    parser.add_argument('--input_edge_dim', type=int, default=2)
    parser.add_argument('--hidden_node_dim', type=int, default=64)
    parser.add_argument('--hidden_edge_dim', type=int, default=16)
    parser.add_argument('--conv_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden_size', type=int, default=128)
    parser.add_argument('--attention_size', type=int, default=128)

    return parser.parse_args()
