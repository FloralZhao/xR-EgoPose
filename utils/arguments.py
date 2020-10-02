import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument('-b', '--batch_size', default=16, help='batch-size', type=int)
    parser.add_argument('-e', '--epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('-opt', '--optimizer', default='adam', type=str, help='optimizer')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--training_type', default='finetune', type=str) # 'Finetune', 'Train2d', 'Train3d'
    parser.add_argument('--load_model', help='the path of the checkpoint to load', type=str)  # default is None
    parser.add_argument('--load_2d_model', help='the path of the checkpoint to load 2D pose detector model', type=str)  # default is None
    parser.add_argument('--load_3d_model', help='the path of the checkpoint to load 3D pose detector model', type=str)  # default is None
    parser.add_argument('--lambda_2d', default=1, help='the weight of the 2d heatmap loss when training 2d and 3d together', type=float)  # default is None
    args = parser.parse_args()

    return args

