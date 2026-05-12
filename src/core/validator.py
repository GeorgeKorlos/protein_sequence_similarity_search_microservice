from src.core.exceptions import (
    SequenceValidationError,
    INVALID_SEQUENCE,
    SEQUENCE_TOO_LONG,
    BATCH_TOO_LARGE,
    PAYLOAD_TOO_LARGE,
)


class SequenceValidator:

    ALLOWED_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYX")
    MAX_SEQUENCE_LENGTH = 1024

    def __init__(self, max_batch_size: int, max_payload_size: int):
        self.max_batch_size = max_batch_size
        self.max_payload_size = max_payload_size

    def validate(self, sequence: str) -> str:
        if not isinstance(sequence, str):
            raise SequenceValidationError(
                INVALID_SEQUENCE,
                "Sequence must be a string",
            )

        sequence = sequence.strip().upper()

        if not sequence:
            raise SequenceValidationError(
                INVALID_SEQUENCE,
                "Sequence cannot be empty",
            )

        if len(sequence) > self.MAX_SEQUENCE_LENGTH:
            raise SequenceValidationError(
                SEQUENCE_TOO_LONG,
                f"Sequence exceeds {self.MAX_SEQUENCE_LENGTH} amino acids",
            )

        if not all(char in self.ALLOWED_CHARS for char in sequence):
            raise SequenceValidationError(
                INVALID_SEQUENCE,
                "Sequence contains invalid amino acid characters",
            )

        return sequence

    def validate_batch(self, sequences: list[str]) -> list[str]:
        if len(sequences) > self.max_batch_size:
            raise SequenceValidationError(
                BATCH_TOO_LARGE, f"Batch exceeds {self.max_batch_size} size"
            )
        if sum(len(sequence) for sequence in sequences) > self.max_payload_size:
            raise SequenceValidationError(
                PAYLOAD_TOO_LARGE, f"Payload exceeds {self.max_payload_size} characters"
            )

        validated_sequences = []

        for index, sequence in enumerate(sequences):
            try:
                validated_sequence = self.validate(sequence)
                validated_sequences.append(validated_sequence)

            except SequenceValidationError as e:
                raise SequenceValidationError(
                    e.error_code,
                    f"Sequence at index {index} failed validation: {e.message}",
                ) from e

        return validated_sequences
