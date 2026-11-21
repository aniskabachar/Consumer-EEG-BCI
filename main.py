"""
EEG BCI Project — Main Entry Point
=====================================
Run this file to execute any stage of the pipeline.

Usage:
  python main.py preprocess  --data_dir ./SEED-VIG  --output_dir ./processed_data
  python main.py train       --data_path ./processed_data/seed_vig_5ch_processed.npz
  python main.py evaluate    --data_path ./processed_data/... --models_dir ./saved_models
  python main.py demo        --model ./saved_models/fold_1_subject_0.pth
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so all submodules resolve correctly
# regardless of what directory you run main.py from
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_preprocess(args):
    from data_processing.preprocessor import process_all_subjects
    process_all_subjects(
        seed_vig_root=args.data_dir,
        output_path=args.output_dir,
        norm=args.norm,
        calib_seconds=args.calib_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )


def cmd_train(args):
    from training.trainer import train
    train(
        data_path=args.data_path,
        save_dir=args.save_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        max_folds=args.max_folds,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        n_val_subjects=args.n_val_subjects,
        patience=args.patience,
    )


def cmd_evaluate(args):
    import torch
    from evaluation.evaluate import evaluate_loso, noise_robustness_test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_loso(args.data_path, args.models_dir, device=device)
    if args.noise_test:
        noise_robustness_test(args.data_path, args.models_dir, device=device)


def cmd_demo(args):
    from deployment.realtime_inference import BCIInferenceEngine
    engine = BCIInferenceEngine(model_path=args.model, n_channels=5)
    engine.start_simulation_mode()


def main():
    p = argparse.ArgumentParser(description='EEG BCI Project')
    sub = p.add_subparsers(dest='command')

    # Preprocess
    pp = sub.add_parser('preprocess')
    pp.add_argument('--data_dir',   default='./SEED-VIG')
    pp.add_argument('--output_dir', default='./processed_data')
    pp.add_argument('--norm', choices=['epoch', 'subject'], default='epoch',
                    help="epoch: z-score each 2 s epoch; subject: z-score each recording "
                         "from its first --calib_seconds (saved as seed_vig_5ch_subjnorm.npz)")
    pp.add_argument('--calib_seconds', type=int, default=300,
                    help='Calibration length for --norm subject')
    pp.add_argument('--window_seconds', type=float, default=2.0,
                    help='Epoch length in seconds (non-default adds _win<N>s to the file name)')
    pp.add_argument('--step_seconds', type=float, default=None,
                    help='Epoch step in seconds (default: half the window = 50%% overlap)')

    # Train
    tr = sub.add_parser('train')
    tr.add_argument('--data_path', default='./processed_data/seed_vig_5ch_processed.npz')
    tr.add_argument('--save_dir',  default='./saved_models')
    tr.add_argument('--epochs',    type=int, default=80)
    tr.add_argument('--batch_size',type=int, default=64)
    tr.add_argument('--max_folds', type=int, default=None,
                    help='Limit number of LOSO folds to run (for quick testing)')
    tr.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader worker processes')
    tr.add_argument('--no_amp', action='store_true',
                    help='Disable CUDA mixed precision training')
    tr.add_argument('--n_val_subjects', type=int, default=3,
                    help='Training subjects held out per fold for epoch selection / early stopping')
    tr.add_argument('--patience', type=int, default=25,
                    help='Early stopping: epochs without validation improvement')

    # Evaluate
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data_path',  required=True)
    ev.add_argument('--models_dir', default='./saved_models')
    ev.add_argument('--noise_test', action='store_true',
                    help='Also run the per-fold noise robustness test')

    # Demo
    dm = sub.add_parser('demo')
    dm.add_argument('--model', required=True)

    args = p.parse_args()
    {
        'preprocess': cmd_preprocess,
        'train':      cmd_train,
        'evaluate':   cmd_evaluate,
        'demo':       cmd_demo,
    }.get(args.command, lambda _: p.print_help())(args)


if __name__ == '__main__':
    main()
