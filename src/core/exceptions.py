class IndexNotReadyError(Exception):
    pass


class ValidationError(Exception):

    def __init__(self, error_code: str, message: str | None = None):
        self.error_code = error_code
        self.message = message or error_code
        super().__init__(self.message)


class SequenceValidationError(ValidationError):
    """Raised when sequence validation fails."""


INVALID_SEQUENCE = "INVALID_SEQUENCE"
SEQUENCE_TOO_LONG = "SEQUENCE_TOO_LONG"
BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
