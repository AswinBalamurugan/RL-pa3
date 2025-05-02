from src.smdp import run_smdp
from src.intraoptions import run_intraoptions
import logging
import os


def setup_logging():
    logging.basicConfig(filename="logs/experiment.log", level=logging.INFO)


def main():
    setup_logging()
    logging.info("Starting SMDP experiment")
    run_smdp()
    logging.info("Starting Intra-Options experiment")
    run_intraoptions()


if __name__ == "__main__":
    main()
