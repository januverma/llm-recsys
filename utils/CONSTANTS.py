DATA_PATH = "./data"
MAX_MOVIE_SEQ_LENGTH = 10
SASREC_CONFIG = {
    "BATCH_SIZE": 16,
    "HIDDEN_DIM": 64,
    "NUM_HEADS": 2,
    "NUM_BLOCKS": 2,
    "DROPOUT": 0.2,
    "LR": 1e-3,
    "EPOCHS": 10
}
PROMPT = (
        "## Instruction: \nBelow is a user's history of likes and dislikes for movies, followed by a target movie."
        " Predict whether the user will like the target movie or not."
        " Only output the prediction as \\boxed{Yes} or \\boxed{No}.\n\n"
        "## User’s history:\n"
)