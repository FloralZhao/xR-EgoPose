import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument('-b', '--batch_size', default=16, help='batch-size', type=int)
    parser.add_argument('-e', '--epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('-opt', '--optimizer', default='adam', type=str, help='optimizer')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--training_type', default='finetune', type=str) # 'Finetune', 'Train2d', 'Train3d', 'Demo'
    parser.add_argument('--step_size', default=10, help='step size for StepLR scheduler', type=int)
    parser.add_argument('--batch_norm', default=False, help='whether use batch normalization or not in autoencoder', type=bool)
    parser.add_argument('--decoder_activation', default=True, help='whether use activation in decoder branch 1', type=bool)

    # ================= load model ======================
    parser.add_argument('--load_model', help='the path of the checkpoint to load', type=str)  # default is None
    parser.add_argument('--load_2d_model', help='the path of the checkpoint to load 2D pose detector model', type=str)  # default is None
    parser.add_argument('--load_3d_model', help='the path of the checkpoint to load 3D pose detector model', type=str)  # default is None

    # ================= loss weight ======================
    parser.add_argument('--lambda_2d', default=1, help='the weight of the 2d heatmap loss when training 2d and 3d together', type=float)
    parser.add_argument('--lambda_recon', default=0.001, help='the weight of heatmap reconstruction loss', type=float)
    parser.add_argument('--lambda_3d', default=0.1, help='the weight of 3d loss', type=float)
    parser.add_argument('--lambda_cos', default=0.01, help='the weight of cosine similarity loss', type=float)
    parser.add_argument('--lambda_len', default=0.5, help='the weight of limb lenght loss', type=float)
    args = parser.parse_args()

    return args

