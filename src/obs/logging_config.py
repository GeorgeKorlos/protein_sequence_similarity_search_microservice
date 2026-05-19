import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "route": getattr(record, "route", None),
            "status_code": getattr(record, "status_code", None),
            "latency_ms": getattr(record, "latency_ms", None),
            "model_version": getattr(record, "model_version", None),
            "index_version": getattr(record, "index_version", None),
            "batch_size": getattr(record, "batch_size", None),
            "seq_len_min": getattr(record, "seq_len_min", None),
            "seq_len_mean": getattr(record, "seq_len_mean", None),
            "seq_len_max": getattr(record, "seq_len_max", None),
            "error_code": getattr(record, "error_code", None),
        }
        if record.exc_info:
            log_dict["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_dict)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
