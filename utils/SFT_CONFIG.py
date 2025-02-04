SFT_PROMPT = (
        "## Instruction: \nBelow is a user's history of likes and dislikes for movies, followed by a target movie."
        " Predict whether the user will like the target movie or not."
        " Only output the prediction as Yes or No.\n\n"
        "## User’s history:\n"
)
USER_PROFILE_PROMPT = (
      "## Instruction: \nBased on a user's profile and viewing history, predict whether the user will like the target movie or not.\n"
      "Only output the prediction as Yes or No.\n\n"
      "## User Profile:\n"
)
MODEL_NAME = "unsloth/Qwen2.5-0.5B"
LORA_CONFIG = {
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None
}
