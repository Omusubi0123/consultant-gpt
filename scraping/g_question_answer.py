import os
import re
from time import sleep
from typing import Any, Literal
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from scraping.g_category_info import base_url, category_dict


def get_question_answer(
    url: str,
    max_retry: int = 3,
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
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

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
            sleep(error_sleep)
    return None, None, None


def get_category_question_answer(
    category_dict: dict[str, Any],
    csv_path: str = "data/goo_question/{}_questions.csv",
    save_file: str = "data/goo_q_n_a/{}.jsonl",
):
    """9つのカテゴリそれぞれについて、質問一覧ページから質問のURLを取得し、CSVファイルに保存する

    Args:
        driver (webdriver.Chrome): chrome driver
        category_dict (dict): 9つのカテゴリに関する情報を格納した辞書
        base_url (dict): 質問一覧ページのURL
    """
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    for category in category_dict.keys():
        df = pd.read_csv(csv_path.format(category))

        q_a_df = pd.DataFrame(columns=["q_id", "q_title", "q_text", "a_texts"])
        for i, url in enumerate(tqdm(df["url"], total=len(df))):
            question_title, question_text, answer_texts = get_question_answer(url)
            question_id = re.search(r"/qa/(\d+)", url).group(1)
            if question_title is not None:
                q_a_dict = {
                    "q_id": question_id,
                    "q_title": question_title,
                    "q_text": question_text,
                    "a_texts": answer_texts,
                }
                q_a_df = pd.concat(
                    [q_a_df, pd.DataFrame([q_a_dict])], ignore_index=True
                )

                if i % 100 == 0:
                    q_a_df.to_json(
                        save_file.format(category),
                        orient="records",
                        lines=True,
                        force_ascii=False,
                    )
            else:
                print(f"Failed to get question and answer from {question_id}")
        q_a_df.to_json(
            save_file.format(category), orient="records", lines=True, force_ascii=False
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

    get_category_question_answer(driver, category_dict)

    driver.quit()
