from src import secret_fn
from src.app import ARBApp
from src.data_model import Composition


def main():
    composition = Composition.parse_tws_composition(secret_fn("composition.txt"))
    app = ARBApp(composition)
    app.execute()


if __name__ == "__main__":
    main()
