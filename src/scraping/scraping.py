# imports
import requests
from bs4 import BeautifulSoup


def main():
    # Get all pages links
    with open("pages.txt") as file:
        pages_urls = file.read().split('\n')

    # Start scrap images
    i = 0
    for page_url in pages_urls:
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, "html.parser")
        items = soup.find_all("img")
        for item in items:
            # check if the image .jpg
            print("Item", item['src'])
            if item['src'][-3:] == 'jpg':
                print(item)
                img_data = requests.get(item['src']).content
                with open(f"../data/{i}.jpg", 'wb') as handler:
                    handler.write(img_data)
                print(f"Image {i} scrapped successfully and saved to data.")
                i += 1


if __name__ == '__main__':
    print("Scrapping started.")
    main()
    print("Scrapping finished.")
