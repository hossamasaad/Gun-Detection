import os


def main():
    # Get all image in data folder
    path = "../../data/"
    images = os.listdir(path=path)

    for i, image in enumerate(images):
        src = path + "/" + image
        dst = path + "/" + f"{i}.jpg"
        os.rename(src=src, dst=dst)


if __name__ == '__main__':
    main()