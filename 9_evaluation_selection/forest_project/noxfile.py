"""Nox sessions."""
import nox
from nox.sessions import Session


nox.options.sessions = ["tests", "flake8"]
locations = "src", "noxfile.py"


@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest")


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run flake8."""
    session.run("poetry", "install", external=True)
    with session.chdir("./src/forest_project/"):
        session.run("flake8")

