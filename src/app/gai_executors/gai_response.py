import os
from typing import Dict, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.app.setting.constant import PROMPT_GENAI_RESPONSE, CUST_DESC
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import pandas as pd
from time import time

class GenAIResponse:

    def __init__(self, sessionId, customerId):
        self.sessionId = sessionId
        self.customerId = customerId
        self.tone = self._set_tone()
        self.chain = self._create_chain_response()
        self.info_df = pd.read_excel(os.environ['INFO_PATH'])
        self.add_info = self._get_info(self.info_df, customerId)

    def _create_chain_response(self):

        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_GENAI_RESPONSE)
        parser = StrOutputParser()

        chain = prompt | model | parser
        return chain

    def _set_tone(self):
        if self.customerId == "A":
            return "高端VIP用戶"
        elif self.customerId == "B":
            return "重點關心客戶"
        else:
            return "一般用戶"

    def _get_info(self, df, customer_id:'str'):
        df_tmp = df.loc[df.customer_id ==customer_id,'定義']
        add_info = str([ix for ix in df_tmp.values])
        return  add_info

    def generate_answer(
        self,
        tid: str,
        message: str,
        consumptionNumber: str,
        totalAmount: str,
        storeName: List,
        categoryName: List,
        **kwargs,
    ) -> Dict:

        response = {}
        response["genAI"] = {"message": None}
        response["template"] = {"tid": None, "blockReason": None}

        try:
            if "被阻擋" in (
                gen_ai_message := self.chain.invoke(
                    {
                        "message": message,
                        "consumptionNumber": consumptionNumber,
                        "totalAmount": totalAmount,
                        "storeName": storeName,
                        "categoryName": categoryName,
                        "tone": self.tone,
                        "desc": CUST_DESC.get(self.tone),
                        "add_info":  self.add_info,
                        # "add_info":  'foo',
                    }
                )
            ):
                response["template"]["tid"] = "98"
                response["template"]["blockReason"] = gen_ai_message

            else:

                response["genAI"]["message"] = gen_ai_message
                response["template"]["tid"] = tid
                response["template"]["blockReason"] = None

        except Exception as e:
            response["template"]["tid"] = "98"
            response["template"]["blockReason"] = str(e)

        finally:
            response["sessionId"] = self.sessionId
            response["customerId"] = self.customerId

        return response


if __name__ == "__main__":

    def test():
        from dotenv import load_dotenv

        load_dotenv(override=True)
        from uuid import uuid4

        sessionId = str(uuid4())
        customerId = "C"
        message = "我過去一年在蝦皮的消費紀錄"
        consumptionNumber = "50"
        totalAmount = "10000"
        storeName = "蝦皮"
        categoryName = None
        tid = "B"
        genai_response = GenAIResponse(
            sessionId=sessionId,
            customerId=customerId,
        )

        response = genai_response.generate_answer(
            message=message,
            tid=tid,
            consumptionNumber=consumptionNumber,
            totalAmount=totalAmount,
            storeName=storeName,
            categoryName=categoryName,
        )
        # print(f"message :{message}")
        # print(response)
    for _ in range(5):
        stime = time()
        test()
        print(time() - stime)
