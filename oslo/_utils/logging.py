import logging
from pathlib import Path
from typing import Union, List
from rich.logging import RichHandler
from oslo.torch._context.parallel_mode import ParallelMode

_FORMAT = "OSLO - %(name)s - %(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=[RichHandler()])


class DistributedLogger:
    _instances = {}

    @staticmethod
    def get_instance(name: str):
        if name in DistributedLogger._instances:
            return DistributedLogger._instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name: str):
        if name in DistributedLogger._instances:
            raise Exception(
                "Logger with the same name has been created."
                "You should use DistributedLogger.get_instance()"
            )
        else:
            self._name = name
            self._logger = logging.getLogger(name)
            DistributedLogger._instances[name] = self

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in [
            "INFO",
            "DEBUG",
            "WARNING",
            "ERROR",
        ], "found invalid logging level."

    def set_level(self, level: str):
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(
        self,
        path: Union[str, Path],
        mode: str = "a",
        level: str = "INFO",
        suffix: str = None,
    ):
        from oslo.torch._context.parallel_context import ParallelContext

        assert isinstance(
            path, (str, Path)
        ), f"expected argument path to be type str or Path, but got {type(path)}"
        self._check_valid_logging_level(level)

        if isinstance(path, str):
            path = Path(path)

        # create log directory
        path.mkdir(parents=True, exist_ok=True)

        # set the default file name if path is a directory
        if not ParallelContext.get_instance().is_initialized(ParallelMode.GLOBAL):
            rank = 0
        else:
            rank = ParallelContext.get_instance().get_global_rank()

        if suffix is not None:
            log_file_name = f"rank_{rank}_{suffix}.log"
        else:
            log_file_name = f"rank_{rank}.log"
        path = path.joinpath(log_file_name)

        # add file handler
        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(_FORMAT)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _log(
        self,
        level,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: list = None,
    ):
        from oslo.torch._context.parallel_context import ParallelContext

        if ranks is None:
            getattr(self._logger, level)(message)
        else:

            local_rank = ParallelContext.get_instance().get_local_rank(parallel_mode)
            if local_rank in ranks:
                getattr(self._logger, level)(message)

    def info(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: list = None,
    ):
        self._log("info", message, parallel_mode, ranks)

    def warning(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: list = None,
    ):
        self._log("warning", message, parallel_mode, ranks)

    def debug(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: list = None,
    ):
        self._log("debug", message, parallel_mode, ranks)

    def error(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: list = None,
    ):
        self._log("error", message, parallel_mode, ranks)


def get_dist_logger(name="OSLO"):
    return DistributedLogger.get_instance(name=name)


def disable_existing_loggers(except_loggers: List[str] = ["OSLO"]):
    for log_name in logging.Logger.manager.loggerDict.keys():
        if log_name not in except_loggers:
            logging.getLogger(log_name).setLevel(logging.WARNING)
