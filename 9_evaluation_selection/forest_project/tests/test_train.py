import pytest

from click.testing import CliRunner
from forest_project.train import train
from forest_project.model import MODEL_RFOREST


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_test_split_ratio(
    runner: CliRunner
) -> None:
    """It Pass when test split ratio is less than 1."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            0.8,
        ],
    )
    assert result.exit_code == 0

def test_error_for_invalid_test_split_ratio(
    runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            42,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output
    
def test_model(
    runner: CliRunner
) -> None:
    """It Pass when model got the defined value."""
    result = runner.invoke(
        train,
        [
            "--model",
            MODEL_RFOREST,
        ],
    )
    assert result.exit_code == 0
    
def test_error_for_model(
        runner: CliRunner
    ) -> None:
        """It Fail when model got an unexpected value."""
        result = runner.invoke(
            train,
            [
                "--model",
                'test',
            ],
        )
        assert result.exit_code == 2

def test_empty_train(
        runner: CliRunner
    ) -> None:
        """It Fail when model got an unexpected value."""
        result = runner.invoke(
            train,
        )
        assert result.exit_code == 0

def test_help_train(
        runner: CliRunner
    ) -> None:
        result = runner.invoke(train, ['--help'])
        assert result.exit_code == 0
