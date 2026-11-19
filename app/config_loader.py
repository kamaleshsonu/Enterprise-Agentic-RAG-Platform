
from pathlib import Path
import yaml


# Locate the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the YAML configuration file
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    """Load application settings from the YAML configuration file."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    config = load_config()
    print(config)
