from analyze import analyze
import pathlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


dir_path = pathlib.Path("./multidata")


def func(id):
    return analyze(id, dir_path)


def main():
    ids = ["0000"] * 10

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(func, tqdm(ids)))

    print(results)
    # save result as json
    with open(dir_path.joinpath("result.json"), "w") as f:
        f.write(str(results))


main()
