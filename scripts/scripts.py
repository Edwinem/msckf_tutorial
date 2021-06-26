from subprocess import check_call


def format() -> None:
    check_call(["isort", "."], )
    check_call(["yapf", "-i", "-r", "--style=style.yapf", "./"], )

def generate_setup_py() -> None:
    f = open("setup.py", "w")
    check_call(["poetry2setup"],stdout=f)

def reformat() -> None:
    check_call(["black", "src/", "tests/"])


def lint() -> None:
    check_call(["flake8", "src/", "tests/"])
    check_call(["mypy", "src/backend/", "tests/"])


def start() -> None:
    check_call(["python", "src/backend/run.py"])


def test() -> None:
    check_call(["pytest", "tests/"])
