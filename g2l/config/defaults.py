import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "G2L"
_C.MODEL.MIXW=None
_C.MODEL.MIX=[]
_C.MODEL.USE_FEAT=False
_C.MODEL.USE_FEAT_BCE=False
_C.MODEL.USE_FEAT_CL=False
_C.MODEL.ADD_TIME_BCE=False
_C.MODEL.ADD_TIME_CL=False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""
_C.DATASETS.THRESHOLD=0.0
_C.DATASETS.DENSE_NUM=10
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.G2L = CN()
_C.MODEL.G2L.NUM_CLIPS = 128
_C.MODEL.G2L.JOINT_SPACE_SIZE = 256

_C.MODEL.G2L.FEATPOOL = CN()
_C.MODEL.G2L.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.G2L.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.G2L.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.G2L.FEAT2D = CN()
_C.MODEL.G2L.FEAT2D.NAME = "pool"
_C.MODEL.G2L.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.G2L.TEXT_ENCODER = CN()
_C.MODEL.G2L.TEXT_ENCODER.NAME = 'BERT'
_C.MODEL.G2L.TEXT_ENCODER.AGGREGATION = 'avg'
_C.MODEL.G2L.TEXT_ENCODER.NORM_MODE = "layernorm"

_C.MODEL.G2L.PREDICTOR = CN() 
_C.MODEL.G2L.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.G2L.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.G2L.PREDICTOR.NUM_STACK_LAYERS = 8


_C.MODEL.G2L.LOSS = CN()
_C.MODEL.G2L.LOSS.MIN_IOU = 0.3
_C.MODEL.G2L.LOSS.MAX_IOU = 0.7
_C.MODEL.G2L.LOSS.BCE_WEIGHT = 1.0
_C.MODEL.G2L.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 1
_C.MODEL.G2L.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.G2L.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.G2L.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.G2L.LOSS.CONTRASTIVE_WEIGHT = 0.05
_C.MODEL.G2L.LOSS.AS_CL_WEIGHT=1.0
_C.MODEL.G2L.LOSS.TAU_VIDEO = 0.2
_C.MODEL.G2L.LOSS.TAU_SENT = 0.2
_C.MODEL.G2L.LOSS.MARGIN = 0.2
_C.MODEL.G2L.LOSS.DENSE_NEG=False
_C.MODEL.G2L.LOSS.AS_CL=False
_C.MODEL.G2L.LOSS.GEO_K = 5
_C.MODEL.G2L.LOSS.GEO_TOPK = 2
_C.MODEL.G2L.LOSS.GEO_W = 1.0
_C.MODEL.G2L.LOSS.HARD_GEO_W = 1.0
_C.MODEL.G2L.LOSS.SHAPLEY_W = 0.001
_C.MODEL.G2L.LOSS.SHAPLEY_NUM = 5
_C.MODEL.G2L.LOSS.SHAPLEY = True
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0
_C.SOLVER.LR_SCHEDULER="multistep"
_C.SOLVER.WARMUP_EPOCHS=0
_C.SOLVER.MAX_NORM=5

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
