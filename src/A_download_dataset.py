import os
import kagglehub


def download_imdb_dataset():
    print("\n")

    raw_data_path = "../data/raw_dataset"

    if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
        os.makedirs(raw_data_path, exist_ok=True)

        custom_cache_directory = os.path.normpath(raw_data_path)
        os.environ["KAGGLEHUB_CACHE"] = custom_cache_directory

        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

        default_cache_directory = os.path.normpath(os.path.expanduser("~/.cache/kagglehub"))
        os.environ["KAGGLEHUB_CACHE"] = default_cache_directory

        print(f"{'\033[92m'}Dataset downloaded successfully in:{'\033[0m'}", os.path.normpath(path))
    else:
        print(f"{'\033[93m'}Dataset has already been downloaded in:{'\033[0m'}", raw_data_path)


if __name__ == "__main__":
    download_imdb_dataset()
