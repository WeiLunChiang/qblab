from typing import Optional
from nemoguardrails.actions import action
import re
import pandas as pd
import os



proprietary_terms = pd.read_excel(os.environ["PROPRIETARY_FILE"], sheet_name="Sheet1")
proprietary_terms = list(proprietary_terms["keyword"])


@action(is_system_action=True)
async def check_blocked_terms_input(context: Optional[dict] = None):

    user_message = context.get("user_message")

    # 判斷關鍵字========================================================
    # proprietary_terms = list(proprietary_terms['keyword'])

    for term in proprietary_terms:
        if term in user_message.lower():
            print(term)
            return True

    # 判斷程式碼========================================================
    code_pattern_dict = [
        r"\b(for|if|def|import|lambda|yield|class)\b",
    ]

    for pattern in code_pattern_dict:
        matches = re.findall(pattern, user_message.lower())
        if (len(set(matches)) >= 2) and (len(user_message.lower()) >= 1000):
            print(pattern)
            return True

    # 判斷數字組合========================================================
    pattern_dict = [
        # 英文字母後包含8~9碼數字身分證號碼
        r"[A-Za-z]([\s\-_.,]*)\d([\s\-_.,]*\d){8,9}",
        # 可能有任意分隔的12碼以上數字(帳號、健保卡號、信用卡號)
        r"(\d[\s\-_.,]*){12}",
        # 8~10碼數字(手機號碼)
        # r"((09[\s\-_.,]*){1}(\d[\s\-_.,]*){8}|(9[\s\-_.,]*){1}(\d[\s\-_.,]*){8}|(\d[\s\-_.,]*){8})",
        r"(09\d{8}|9\d{8}|\d{8})",

        # 1~2碼英文+6~9碼數字(居留證號碼、護照號碼)
        r"[A-Za-z]{1,2}[\s\-_.,]*[0-9]{6,9}",
    ]

    for pattern in pattern_dict:
        if re.search(pattern, user_message.lower()):
            # print(user_message.lower())
            print(pattern)
            return True

    return False


@action(is_system_action=True)
async def check_blocked_terms_output(context: Optional[dict] = None):
    bot_response = context.get("bot_message")


    # 判斷關鍵字========================================================
    # proprietary_terms = list(proprietary_terms['keyword'])

    for term in proprietary_terms:
        if term in bot_response.lower():
            print(term)
            return True

    # 判斷程式碼========================================================
    code_pattern_dict = [
        r"\b(for|if|def|import|lambda|yield|class)\b",
    ]

    for pattern in code_pattern_dict:
        matches = re.findall(pattern, bot_response.lower())
        if (len(set(matches)) >= 2) and (len(bot_response.lower()) >= 1000):
            print(pattern)
            return True

    # 判斷數字組合========================================================
    pattern_dict = [
        # 英文字母後包含8~9碼數字身分證號碼
        r"[A-Za-z]([\s\-_.,]*)\d([\s\-_.,]*\d){8,9}",
        # 可能有任意分隔的12碼以上數字(帳號、健保卡號、信用卡號)
        r"(\d[\s\-_.,]*){12}",
        # 8~10碼數字(手機號碼)
        r"(09\d{8}|9\d{8}|\d{8})",

        # 1~2碼英文+6~9碼數字(居留證號碼、護照號碼)
        r"[A-Za-z]{1,2}[\s\-_.,]*[0-9]{6,9}",
    ]

    for pattern in pattern_dict:
        if re.search(pattern, bot_response.lower()):
            print(pattern)
            return True

    return False