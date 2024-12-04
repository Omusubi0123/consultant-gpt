import os
from typing import Literal
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraping.g_category_info import base_url, category_dict


def get_question_urls(
    driver: webdriver.Chrome,
    category: str,
    base_url: str,
    category_id: int,
    page_num: int,
    key: Literal["new", "best", "ranking", "ans_num"],
    save_path: str = "data/goo_question/{category}_questions.csv",
):
    save_path = save_path.format(category=category)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        question_df = pd.DataFrame(columns=["category", "title", "url", "key", "page"])
    else:
        question_df = pd.read_csv(save_path)

    for page in range(1, page_num + 1):
        print(f"{category} {key} page {page}/{page_num}")
        try:
            url = base_url.format(category_id=category_id, page=page)

            # ページの読み込み
            driver.get(url)

            # ページが完全に読み込まれるまで待機
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "qalistItem"))
            )

            # ページのソースを取得
            page_source = driver.page_source

            # BeautifulSoupでHTMLを解析
            soup = BeautifulSoup(page_source, "html.parser")

            # <li class="qalistItem">要素を全て取得
            qalist_items = soup.find_all("li", class_="qalistItem")

            # URLを格納するリスト
            urls = []
            titles = []
            # 各qalistItem要素から<a>タグのURLを取得
            for item in qalist_items:
                a_tag = item.find("a", attrs={"data-osccid": "qa"})
                if a_tag:
                    titles.append(a_tag.text.strip())
                    urls.append(urljoin(url, a_tag["href"]))

            # 取得したURLをDataFrameに格納
            df = pd.DataFrame(
                {
                    "category": [category] * len(urls),
                    "title": titles,
                    "url": urls,
                    "key": [key] * len(urls),
                    "page": [page] * len(urls),
                }
            )
            question_df = pd.concat([question_df, df], ignore_index=True)
        except Exception as e:
            print(e)
            
    question_df.to_csv(save_path, index=False)


def get_category_questions(driver, category_dict, base_url):
    for category, value in category_dict.items():
        category_id = value["category_id"]
        page_num = value["page_num"]

        for key in page_num.keys():
            get_question_urls(
                driver, category, base_url[key], category_id, page_num[key], key
            )


if __name__ == "__main__":
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service("/usr/local/bin/chromedriver-linux64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    get_category_questions(driver, category_dict, base_url)

    driver.quit()
