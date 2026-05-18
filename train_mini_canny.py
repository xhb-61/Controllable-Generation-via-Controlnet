import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ControlNet on the mini Canny task.")
    parser.add_argument("--data-root", default="./training/mini_canny")
    parser.add_argument("--resume-path", default="./models/control_sd15_canny.pth")
    parser.add_argument("--model-config", default="./models/cldm_v15.yaml")
    parser.add_argument("--output-dir", default="./training/mini_canny_runs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--logger-freq", type=int, default=100)
    parser.add_argument("--save-every-n-steps", type=int, default=250)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-pl-ckpt", default=None)
    parser.add_argument("--only-mid-control", action="store_true")
    parser.add_argument("--unlock-sd", dest="sd_locked", action="store_false")
    parser.set_defaults(sd_locked=True)
    return parser.parse_args()


def main():
    args = parse_args()
    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Missing {resume_path}. Download the Canny ControlNet checkpoint before training."
        )

    import share  # noqa: F401
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from cldm.logger import ImageLogger
    from cldm.model import create_model, load_state_dict
    from mini_canny_dataset import MiniCannyDataset

    pl.seed_everything(args.seed)

    model = create_model(args.model_config).cpu()
    model.load_state_dict(load_state_dict(str(resume_path), location="cpu"))
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    dataset = MiniCannyDataset(args.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="mini-canny-{step:06d}",
        every_n_train_steps=args.save_every_n_steps,
        save_last=True,
        save_top_k=-1,
    )
    image_logger = ImageLogger(batch_frequency=args.logger_freq)

    trainer_kwargs = dict(
        gpus=args.gpus,
        precision=args.precision,
        callbacks=[image_logger, checkpoint_callback],
        default_root_dir=args.output_dir,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    if args.resume_pl_ckpt:
        trainer_kwargs["resume_from_checkpoint"] = args.resume_pl_ckpt

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
