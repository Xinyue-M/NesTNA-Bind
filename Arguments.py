'''
Author: maxy0617 maxy@shanghaitech.edu.cn
Date: 2023-05-14 21:31:34
LastEditors: maxy0617 maxy@shanghaitech.edu.cn
LastEditTime: 2024-06-20 14:00:50
FilePath: \GeoBind-main\Arguments.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse

parser = argparse.ArgumentParser(description="Network parameters")

# Main parameters
parser.add_argument(
    "--taskname", type=str, help="DNA or RNA", choices=['RNA', 'DNA', 'BOTH', 'RNA_our', 'PPI'], default='RNA'
)

parser.add_argument(
    "--taskbase", type=str, help="patch or site", choices=['site', 'patch'], default='patch'
)

parser.add_argument(
    "--use_esm",
    type=bool,
    default=False,
)

parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default="./logs",
    help="Where the log and model save",
)

parser.add_argument(
    "--num_sampling",
    # type=int,
    default=1024,
)

parser.add_argument(
    "--num_neighbors",
    type=int, 
    default=128
)

parser.add_argument(
    "--split_ratio",
    type=float,
    default=0.8
)

parser.add_argument(
    "--loss_type",
    type=str,
    default='vertices', # interface or not
    # choices=['vertice, residue'],
    help="loss type interface if sum of point cloud, others if sum of sites,",
)


parser.add_argument(
    "--input_feature_type",
    type=str,
    nargs='+',
    default=['chem','atomtype', 'geo'], #attention the feature order should be the same with first time train when loaded.
    help="hmm:30, chemi:6, geo:1,",
)

parser.add_argument(
    "--load_model_path",
    type=str,
    default=None,
    help="to load model",
)

parser.add_argument(
    "--start_epoch",
    type=int,
    default=0,
)

parser.add_argument(
    "--n_layers", type=int, default=2, help="Number of convolutional layers"
)

parser.add_argument("--seed", type=int, default=42, help="Random seed")

parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="learning rate",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="max number of epochs",
)

parser.add_argument(
    "--nclasses",
    type=int,
    default=2,
    help="Where the log and model save",
)

parser.add_argument(
    "--emb_dims",
    type=int,
    default=64,
    help="Number of input features (+ 3 xyz coordinates for DGCNNs)",
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
    help="dropout probability",
)
