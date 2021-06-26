from subprocess import check_call


def format() -> None:
    check_call(["isort", "."], )
    check_call(["yapf", "-i", "-r", "--style=style.yapf", "./"], )


def reformat() -> None:
    check_call(["black", "src/", "tests/"])


def lint() -> None:
    check_call(["flake8", "src/", "tests/"])
    check_call(["mypy", "src/backend/", "tests/"])


def start() -> None:
    check_call(["python", "src/backend/run.py"])


def test() -> None:
    check_call(["pytest", "tests/"])
