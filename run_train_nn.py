from nn_model.nn_model import process_nn_model
import pandas as pd

def main(args):
    df = pd.read_csv(args.input_csv)
    process_nn_model(
        df=df,
        scaler_type=args.scaler_type,
        loss=args.loss,
        optimiser=args.optimiser,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        save_path=args.output_path,
        device=args.device,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--input_csv', type=str, help='Path to preprocessed simulation results', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save model', required=True)
    parser.add_argument('--scaler_type', type=str, help='Type of scaler to use', required=False, default="max_abs"),# options=["max_abs", "min_max", "standard", "robust"])
    parser.add_argument('--loss', type=str, help='Loss function to use', required=False, default="mse"),# options=["mse", "mae", "huber"])
    parser.add_argument('--optimiser', type=str, help='Optimiser to use', required=False, default="adam"),# options=["adam", "sgd", "rmsprop"])
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=100)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)
    parser.add_argument('--test_size', type=float, help='Test size', required=False, default=0.3)
    parser.add_argument('--device', type=str, help='Device to use', required=False, default="cpu"),# options=["cpu", "cuda"])
    parser.add_argument('--lr', type=int, help='Learning rate', required=False, default=1e-3)
    parser.add_argument('--momentum', type=int, help='Momentum', required=False, default=0.9)
    parser.add_argument('--weight_decay', type=int, help='Weight decay', required=False, default=None)
    args = parser.parse_args()

    main(args)
