import requests
from bs4 import BeautifulSoup
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=200000, help='id of the start picture')
args = parser.parse_args()
total_number = 10000
number = 0

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
}

def download(url, save_dir):
    global number
    try:
        img_res = requests.get(url, headers=headers, timeout=5, stream=True)
        img_res.raise_for_status()
    except:
        return
    fil_name = os.path.join(save_dir, url.split('/')[-1])
    if os.path.exists(fil_name):
        print("%s already exists!" % fil_name)
        return
    with open(fil_name, "wb") as fil:
        for chunk in img_res.iter_content(chunk_size=1024):
            if chunk:
                fil.write(chunk)
                fil.flush()
        number += 1
    if number % 200 == 0:
        time.sleep(10)

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), 'data/comic_faces/origin')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = args.start
    while number < total_number:
        url = 'http://konachan.net/post/show/%d/' % i
        result = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(result.text, 'html.parser')
        for img in soup.find_all('img', class_='image'):
            target_url = img['src']
            download(target_url, save_dir)
        print("[%d / %d] downloading %d" % (number, total_number, i))
        i += 1

    print("\ntotal_numbers: %d pictures downloaded!" % number )