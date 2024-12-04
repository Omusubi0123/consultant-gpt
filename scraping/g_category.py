from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Chromeドライバーの設定
chrome_options = Options()
chrome_options.add_argument("--headless")
service = Service("/usr/local/bin/chromedriver-linux64/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)

url = "https://oshiete.goo.ne.jp/articles/qa/2549/?sort=2&target=best"

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

# 各qalistItem要素から<a>タグのURLを取得
for item in qalist_items:
    a_tag = item.find("a", attrs={"data-osccid": "qa"})
    if a_tag and "href" in a_tag.attrs:
        # 相対URLを絶対URLに変換
        absolute_url = urljoin(url, a_tag["href"])
        urls.append(absolute_url)

# 取得したURLを表示
for url in urls:
    print(url)

# ブラウザを閉じる
driver.quit()
