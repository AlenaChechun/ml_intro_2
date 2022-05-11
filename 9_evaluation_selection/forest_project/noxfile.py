"""Nox sessions."""

import tempfile
from typing import Any

import nox
from nox.sessions import Session


nox.options.sessions = ["tests", "flake8"]
locations = "src", "noxfile.py"


@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest")


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run flake8."""
    session.run("poetry", "install", external=True)
    with session.chdir("./src/forest_project/"):
        session.run("flake8")

