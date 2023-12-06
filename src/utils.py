import json
import logging
from datetime import datetime
from pathlib import Path, PosixPath

import pyhocon
import transformers

logger = logging.getLogger(__name__)


def init_config(config_path: Path, config_name: str, debug: bool = False):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if "seed" in config:
        transformers.set_seed(config["seed"])

    for x in ["output_path", "hf_tokenized_dataset_path", "log_path", "preds_path"]:
        if x in config:
            config[x] = Path(config[x])
            config[x] /= f"{config_name}"

    if "resume_checkpoint" in config:
        # reuse the same output_path
        config["output_path"] = Path(config["resume_checkpoint"])
    elif not config.get("eval", False):
        # only add new timestamp if we are not resuming from a checkpoint
        config["output_path"] /= f"{timestamp}"
        config["log_path"] /= f"{timestamp}"

    config["output_path"].mkdir(exist_ok=True, parents=True)
    config["log_path"].mkdir(exist_ok=True, parents=True)

    # copy deepspeed config into main config (solely for wandb logging purposes)
    if "deepspeed_config" in config:
        with open(config["deepspeed_config"], "r") as rf:
            deepspeed_config = json.load(rf)
            config["deepspeed_params"] = deepspeed_config

    LOG_LEVEL = logging.DEBUG if debug else logging.INFO
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            config["log_path"] / f"log_{timestamp}.txt",
            mode="w",
        ),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=LOG_LEVEL,
        handlers=handlers,
    )
    transformers_handlers = [
        logging.FileHandler(
            config["log_path"] / f"hf_log_{timestamp}.txt",
            mode="w",
        )
    ]
    transformers.logging.set_verbosity_error()
    transformers.logging.add_handler(transformers_handlers[0])

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    return config


def convert_to_dict(config):
    config_dict = {}
    for k, v in config.items():
        if isinstance(v, pyhocon.ConfigTree):
            config_dict[k] = convert_to_dict(v)
        elif isinstance(v, PosixPath):
            config_dict[k] = str(v)
        else:
            config_dict[k] = v
    return config_dict
