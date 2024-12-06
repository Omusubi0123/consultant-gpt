from typing import Any

import pandas as pd

from scraping.g_category_info import category_dict


def exclude_overlap_question(csv_path: str):
    """重複した質問を除外する

    Args:
        csv_path (str): 質問のURLが格納されたCSVファイルのパス
    """
    question_df = pd.read_csv(csv_path)
    question_df = question_df.drop_duplicates(subset=["url"])
    print(f"Number of questions: {len(question_df)}")
    question_df.to_csv(csv_path, index=False)
    return len(question_df)


def main(category_dict: dict[str, Any]):
    """9つのカテゴリそれぞれについて、質問一覧ページから質問のURLを取得し、CSVファイルに保存する

    Args:
        driver (webdriver.Chrome): chrome driver
        category_dict (dict): 9つのカテゴリに関する情報を格納した辞書
        base_url (dict): 質問一覧ページのURL
    """
    total_question = 0
    for category in category_dict.keys():
        print(f"Category: {category}")
        csv_path = f"data/goo_question/{category}_questions.csv"
        total_question += exclude_overlap_question(csv_path)
    print(f"Total number of questions: {total_question}")


if __name__ == "__main__":
    main(category_dict)
