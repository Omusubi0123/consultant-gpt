import os
from time import sleep
from typing import Any, Literal
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
    max_retry: int = 3,
    timeout: int = 20,
    error_sleep: int = 5,
):
    """指定したカテゴリの質問一覧ページから質問のURLを取得し、CSVファイルに保存する

    Args:
        driver (webdriver.Chrome): chrome driver
        category (str): カテゴリ名
        base_url (str): 質問一覧ページのURL
        category_id (int):  教えてgooのカテゴリID
        page_num (int): 取得するページ数
        key (Literal[&quot;new&quot;, &quot;best&quot;, &quot;ranking&quot;, &quot;ans_num&quot;]): 取得する質問の種類
        save_path (str, optional): 保存先のファイルパス. Defaults to &quot;data/goo_question/{category}_questions.csv&quot;.
        max_retry (int, optional): 最大リトライ回数. Defaults to 3.
        timeout (int, optional): ページ読み込みのタイムアウト時間. Defaults to 20.
        error_sleep (int, optional): エラー時の待機時間. Defaults to 5.
    """
    save_path = save_path.format(category=category)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        question_df = pd.read_csv(save_path)
    else:
        question_df = pd.DataFrame(columns=["category", "title", "url", "key", "page"])

    for page in range(1, page_num + 1):
        print(f"{category} {key} page {page}/{page_num}")

        retry = 0
        while retry < max_retry:
            try:
                url = base_url.format(category_id=category_id, page=page)

                # ページの読み込み
                driver.get(url)

                # ページが完全に読み込まれるまで待機
                element_present = EC.presence_of_element_located(
                    (By.CLASS_NAME, "qalistItem")
                )
                WebDriverWait(driver, timeout).until(element_present)

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
                    a_tag = item.find(
                        "a",
                        attrs={"data-osccid": "qa" if key != "ranking" else "ranking"},
                    )
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
                break
            except Exception as e:
                retry += 1
                print(e)
                sleep(error_sleep)

    question_df.to_csv(save_path, index=False)


def get_category_questions(
    driver: webdriver.Chrome, category_dict: dict[str, Any], base_url: dict[str, str]
):
    """9つのカテゴリそれぞれについて、質問一覧ページから質問のURLを取得し、CSVファイルに保存する

    Args:
        driver (webdriver.Chrome): chrome driver
        category_dict (dict): 9つのカテゴリに関する情報を格納した辞書
        base_url (dict): 質問一覧ページのURL
    """
    for category, value in category_dict.items():
        category_id = value["category_id"]
        page_num = value["page_num"]

        for key in page_num.keys():
            get_question_urls(
                driver, category, base_url[key], category_id, page_num[key], key
            )


if __name__ == "__main__":
    timeout = 20

    # Chromeのオプションを設定
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--dns-prefetch-disable")
    service = Service("/usr/local/bin/chromedriver-linux64/chromedriver")

    # Chromeを起動
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.timeouts.page_load = timeout * 1000

    get_category_questions(driver, category_dict, base_url)

    driver.quit()
