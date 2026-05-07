import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from trainers.config import get_dataset_specified_config
from yacs.config import CfgNode as CN
import torch.multiprocessing as mp
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.mmrl
import trainers.mmadapter
import trainers.cocoop
import trainers.coop
import trainers.promptsrc

import trainers.mmrl_nerp
import trainers.mmadapter_nerp
import trainers.promptsrc_nerp
import trainers.cocoop_nerp

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.train_epoch:
        cfg.OPTIM.MAX_EPOCH = args.train_epoch

    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    cfg.TRAINER.MMRL = CN()
    cfg.TRAINER.MMRL.ALPHA = 0.7
    cfg.TRAINER.MMRL.REG_WEIGHT = 1.0
    cfg.TRAINER.MMRL.REP_LAYERS = []
    cfg.TRAINER.MMRL.REP_DIM = 512
    cfg.TRAINER.MMRL.N_REP_TOKENS = 5  # number of representation tokens per layer
    cfg.TRAINER.MMRL.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TASK = "B2N" #B2N, CD, FS

    cfg.TRAINER.CRL = CN()
    cfg.TRAINER.CRL.ALPHA = 0.7
    cfg.TRAINER.CRL.REG_WEIGHT = 1.0
    cfg.TRAINER.CRL.REP_LAYERS = []
    cfg.TRAINER.CRL.REP_DIM = 512
    cfg.TRAINER.CRL.N_REP_TOKENS = 5  # number of representation tokens per layer
    cfg.TRAINER.CRL.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TEST.TAU = 0.0
    cfg.TEST.THRES = 2.0
    cfg.TEST.RESIDUAL_IMAGE_PRIOR = False
    cfg.TEST.IMAGE_PRIOR_COE = 1.0
    cfg.TEST.RESIDUAL_TEXT_PRIOR = False
    cfg.TEST.TEXT_PRIOR_COE = 1.0
    cfg.TEST.FLIP_BY_PROB = False
    cfg.TEST.FLIP_BY_ENTROPY = False
    cfg.TEST.ENTROPY_THRES = 1.0
    cfg.TEST.FLIP_RULE = "OR"
    cfg.TASK = "B2N" #B2N, CD, FS

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    # cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    
    cfg.TRAINER.COOP.W = 2.0
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.HICROPL = CN()
    cfg.TRAINER.HICROPL.N_CTX = 2 # number of context vectors
    cfg.TRAINER.HICROPL.CROSS_LAYER = 6 # cross layer
    cfg.TRAINER.HICROPL.CTX_INIT = "a photo of a" # initialization words (only for language prompts)
    cfg.TRAINER.HICROPL.PREC = "fp16"
    cfg.TRAINER.HICROPL.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow HICROPL (J=1)
    cfg.TRAINER.HICROPL.TEACHER_NAME = "ViT-L/14"
    cfg.TRAINER.HICROPL.LAMBD = 10.
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


    cfg.TRAINER.MMADAPTER = CN()
    cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT = ""  # initialization words
    cfg.TRAINER.MMADAPTER.PREC = "amp"  # fp16, fp32, amp

    # Put text prompt ctx before transformer 
    cfg.TRAINER.MMADAPTER.ADAPTER_START = 4
    cfg.TRAINER.MMADAPTER.ADAPTER_END = 12
    cfg.TRAINER.MMADAPTER.ADAPTER_DIM = 32
    cfg.TRAINER.MMADAPTER.ADAPTER_SCALE = 0.1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Override dataset specific config
    if cfg.TRAINER.NAME in ["MMRL", "MMRL_NeRP", "PromptSRC_NeRP", "CoCoOp_NeRP", "MultiModalAdapter_NeRP"]:
        cfg.merge_from_list(get_dataset_specified_config(dataset=cfg.DATASET.NAME, trainer=cfg.TRAINER.NAME, task=args.opts[args.opts.index('TASK') + 1], subsample=cfg.DATASET.SUBSAMPLE_CLASSES))
    
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--train-epoch", type=int, help="the max epoch for training model"
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
