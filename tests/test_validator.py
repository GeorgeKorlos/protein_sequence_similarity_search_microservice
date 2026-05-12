import pytest
from src.core.validator import SequenceValidator
from src.core.exceptions import (
    SequenceValidationError,
    INVALID_SEQUENCE,
    SEQUENCE_TOO_LONG,
    BATCH_TOO_LARGE,
    PAYLOAD_TOO_LARGE,
)

validator = SequenceValidator(max_batch_size=3, max_payload_size=100)


def test_lowercase_input_is_uppercased_and_returned_correctly():
    sequence = "acdefghi"
    expected = "ACDEFGHI"
    validated_sequence = validator.validate(sequence)
    assert validated_sequence == expected


def test_empty_string_raises_sequence_validation_error_with_invalid_sequence_code():
    empty = ""
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate(empty)
    assert exc_info.value.error_code == INVALID_SEQUENCE


def test_sequence_length_greater_than_max_length_raises_sequence_too_long():
    long_sequence = "A" * 2000
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate(long_sequence)
    assert exc_info.value.error_code == SEQUENCE_TOO_LONG


def test_sequence_with_invalid_character_raises_invalid_sequence_code():
    sequence = "ABCDEFG"
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate(sequence)
    assert exc_info.value.error_code == INVALID_SEQUENCE


def test_valid_clean_sequence_returns_expected_string():
    sequence = "ACDEFGHIKLMNPQRSTVWYX"
    validated_sequence = validator.validate(sequence)
    assert validated_sequence == sequence


def test_batch_exceeding_max_batch_size_raises_batch_too_large():
    sequences = ["ACDEFGHIK"] * (validator.max_batch_size + 1)
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate_batch(sequences)
    assert exc_info.value.error_code == BATCH_TOO_LARGE


def test_batch_exceeding_max_payload_size_raises_payload_too_large():
    sequences = [
        "A" * 600,
        "C" * 500,
    ]
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate_batch(sequences)
    assert exc_info.value.error_code == PAYLOAD_TOO_LARGE


def test_failing_sequence_in_batch_includes_index_in_error_message():
    sequences = [
        "ACDEFG",
        "INVALID123",
        "QRSTVWY",
    ]
    with pytest.raises(SequenceValidationError) as exc_info:
        validator.validate_batch(sequences)
    assert exc_info.value.error_code == INVALID_SEQUENCE
    assert "index 1" in exc_info.value.message
