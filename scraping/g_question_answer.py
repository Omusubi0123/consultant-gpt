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


def get_question_answer(
    driver: webdriver.Chrome,
    url: str,
    max_retry: int = 3,
    timeout: int = 20,
    error_sleep: int = 5,
):
    """指定したURLにアクセスし、そのページのquestionとすべてのanswerを取得する

    Args:
        driver (webdriver.Chrome): chrome driver
        url (str): 質問ページのURL
        max_retry (int, optional): 最大リトライ回数. Defaults to 3.
        timeout (int, optional): ページ読み込みのタイムアウト時間. Defaults to 20.
        error_sleep (int, optional): エラー時の待機時間. Defaults to 5.
    """
    retry = 0
    while retry < max_retry:
        try:
            url = url

            driver.get(url)

            element_present = EC.presence_of_element_located((By.CLASS_NAME, "a_text"))
            WebDriverWait(driver, timeout).until(element_present)

            page_source = driver.page_source

            soup = BeautifulSoup(page_source, "html.parser")

            question_title = soup.find("div", class_="q_article_info").find("h1").text
            question_text = soup.find("p", class_="q_text").get_text(
                strip=True, separator="\n"
            )
            answer_items = soup.find_all("div", class_="a_text")

            answer_texts = []
            for item in answer_items:
                text = item.get_text(strip=True, separator="\n")
                text = "\n".join(line for line in text.split("\n") if line.strip())
                answer_texts.append(text)

            return question_title, question_text, answer_texts
        except Exception as e:
            retry += 1
            print(e)
    return None, None, None


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
