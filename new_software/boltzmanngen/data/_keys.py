from typing import Final


# == Define allowed keys as constants == #

# === Define keys for classical Neural Networks === #

INPUT_KEY: Final[str] = "x"
OUTPUT_KEY: Final[str] = "y"

FIRST_CHANNEL_KEY: Final[str] = "x1"
SECOND_CHANNEL_KEY: Final[str] = "x2"

FIRST_CHANNEL_OUT_KEY: Final[str] = "f(x1)"
SECOND_CHANNEL_OUT_KEY: Final[str] = "f(x2)"

JACOB_KEY: Final[str] = "J"

Z_TO_X_KEY: Final[str] = "zx"
X_TO_Z_KEY: Final[str] = "xz"
SADDLE_KEY: Final[str] = "saddle"
PATH_KEY: Final[str] = "path"

TRAIN_BY_ENERGY: Final[tuple] = (OUTPUT_KEY, JACOB_KEY, Z_TO_X_KEY)
TRAIN_BY_EXAMPLE: Final[tuple] = (OUTPUT_KEY, JACOB_KEY, X_TO_Z_KEY)
TRAIN_BY_HESSIAN: Final[tuple] = (OUTPUT_KEY, JACOB_KEY, SADDLE_KEY)
TRAIN_BY_PATH: Final[tuple] = (OUTPUT_KEY, JACOB_KEY, PATH_KEY)

# === Define keys for Graph Neural Networks === #