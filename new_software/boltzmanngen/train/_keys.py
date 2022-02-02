from boltzmanngen.data import DataConfig


LOSS_KEY = "total_loss"
VALIDATION = "validation"
TRAIN = "training"

ABBREV = {
    DataConfig.TRAIN_BY_ENERGY: "Jkl",
    DataConfig.TRAIN_BY_EXAMPLE: "Jml",
    DataConfig.TRAIN_BY_HESSIAN: "Jhe",
    LOSS_KEY: "loss",
    VALIDATION: "val",
    TRAIN: "train",
}
