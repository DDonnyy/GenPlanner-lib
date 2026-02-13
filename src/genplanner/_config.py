import sys
from typing import Literal

from loguru import logger


class Config:

    def __init__(self):
        self.logger = logger
        self.roads_width_def = {"high speed highway": 20, "regulated highway": 10, "local road": 5}
        self.point_pool_seed = 42
        self.point_pool_size = 200

    def change_logger_lvl(self, lvl: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"]):
        self.logger.remove()
        self.logger.add(sys.stderr, level=lvl)


config = Config()
config.change_logger_lvl("INFO")
