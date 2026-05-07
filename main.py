import argparse
import yaml
import torch
import train, test #, inference


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Super-resolution Experiment Runner")
    parser.add_argument(
        "--stage", "-s",
        type=str,
        required=True,
        choices=["train", "test", "infer"],
        help="Which stage to run: train | test | infer"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # === Load config and set device ===
    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    print("\n=== Super-Resolution Experiment Parameters ===")
    print(f"Using device: {device}")
    print(f"Config file: {args.config}")
    print(f"Running stage: {args.stage}")
    print(f"Upper Bound Experiment: {not(cfg['data']['smap'])}")
    if args.stage == "train":
        print(f"Logging training process: {cfg['training']['log']}")
    # if cfg["ensemble"]["enabled"]:
    #     print(f"\nModel: {cfg['ensemble']['arch']}")
    else:
        print(f"\nModel: {cfg['model']['arch']}")
    print(f"Coarse Resolution: {cfg['data']['coarse_res']}")
    print(f"Fine Resolution: {cfg['data']['fine_res']}")
    print(f"Scale Factor: {cfg['data']['scale_factor']}")
    print("=============================================\n")

    # === Dispatch to stage ===
    if args.stage == "train":
        train.run(cfg, device)
    elif args.stage == "test":
        test.run(cfg, device)
    else:
        raise ValueError("Unknown stage argument!")


if __name__ == "__main__":
    main()
