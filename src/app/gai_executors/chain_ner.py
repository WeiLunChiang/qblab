import logging
import os
from typing import Dict, List
from datetime import datetime
from functools import partial
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy.engine import Connectable
from src.app.vdb_connector import ChromaDBClient
from src.app.setting.utils_retriever import RetrieveWithScore, get_metadata_runnable
from src.app.setting.constant import PROMPT_COMPLETETION, PROMPT_QUESTION_NER
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import pandas as pd
import re
from time import time


from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

class ChainNer:
    def __init__(
        self,
        sessionId: str,
        customerId: str,
        chromaCollection: str,
        engine: Connectable,
        time: str,
        k: int = 1,
        scoreThreshold: float = 0.2,
        **kwargs,
    ):
        self.time = self._convert_time_format(time)
        self.sessionId = sessionId
        self.customerId = customerId
        self.model = self._create_model()
        self.vectorstore = self._create_vectorstore(chromaCollection)
        self.retriever = self._create_retriever(k, scoreThreshold)
        self.chian_completeion = self._create_chain_completeion(engine)
        self.chain_ner = self._create_chain_ner()
        self.proprietary_terms = self._load_proprietary_terms()

    def _load_proprietary_terms(self) -> List[str]:
        """
        init proprietary keywords from excel file
        """
        
        proprietary_terms = pd.read_excel(os.environ['PROPRIETARY_FILE'])
        logger.info("load proprietary keywords from excel")

        return list(proprietary_terms['keyword'])

    def _check_blocked_terms(self,user_message: str, block_words = "被阻擋") -> str:
        start_time = time()
        # 判斷關鍵字========================================================

        for term in self.proprietary_terms:
            if term in user_message.lower():
                logger.info(f'block by term: {term} \n user message: {user_message}')
                return block_words

        # 判斷程式碼========================================================
        code_pattern_dict = [
            r"\b(for|if|def|import|lambda|yield|class)\b",
        ]

        for pattern in code_pattern_dict:
            matches = re.findall(pattern, user_message.lower())
            if (len(set(matches)) >= 2) and (len(user_message.lower()) >= 1000):
                logger.info(f'block by pattern: {pattern} \n user message: {user_message}')
                return block_words

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
            if re.search(pattern, user_message.lower()):
                logger.info(f'block by pattern: {pattern} \n user message: {user_message}')
                return block_words
        end_time = time()
        print( f'time usage: {end_time - start_time}')
        return user_message

    def search(self, user_input_raw: str, **kwargs) -> Dict:

        response = {}
        template = {}
        template["tid"] = None
        template["blockReason"] = None
        template["startDate"] = None
        template["endDate"] = None
        template["storeName"] = None
        template["categoryName"] = None
        template["message"] = None

        logger.info(f"session_id: {self.sessionId}")
        logger.info(f"Raw user input: {user_input_raw}")
        user_input = None
        try:
            # s_time = time()
            
            if  "被阻擋" in (user_input_raw := self._check_blocked_terms(user_input_raw)): #add for live demo
                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input_raw
                logger.info(f"====================user_input_raw: {user_input_raw} block by guardrails====================")

            elif "被阻擋" in (
                user_input := self.chian_completeion.invoke(
                    {"user_input": user_input_raw},
                    config={"configurable": {"session_id": self.sessionId}},
                )
            ):
                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input_raw
                # self.chain_ner.invoke(user_input)會失敗
                logger.info(f"====================user_input_raw: {user_input_raw} block by guardrails====================")

            elif  "被阻擋" in (user_input := self._check_blocked_terms(user_input)): #add for live demo
                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input
                logger.info(f"====================user_input: {user_input} block by guardrails====================")

            elif "被阻擋" in (result := self.chain_ner.invoke(user_input)):

                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input
                logger.info(f"====================user_input:{user_input} ner result by guardrails====================")

            elif not result["retriever"]["score"]:
                template["tid"] = "99"  # TBD
                template["blockReason"] = "相似度過低"
                logger.info(f"====================user_input: {user_input}  similarity too low, block by vdb====================")

            else:
                # print(s_time - time())
                template["tid"] = str(result["retriever"].get("category", None)[0])
                template["startDate"] = str(result["keys"].get("&start_date", None))
                template["endDate"] = str(result["keys"].get("&end_date", None))
                if "&string1" in result["keys"]:
                    template["storeName"] = [
                        result["keys"].get("&string1")
                    ]
                    if "&string2" in result["keys"]:
                        template["storeName"].append(result["keys"].get("&string2"))

                template["categoryName"] = [result["keys"].get("&string")] if "&string" in result["keys"] else None
                template["message"] = str(user_input)


                modify_query = result["keys"].get("modify_query")
                page_content = result["retriever"]["data"][0].page_content
                metadata = result["retriever"]["data"][0].metadata

                logger.info(f"VDB modify_query: {modify_query}")
                logger.info(f"VDB page_content: {page_content}")
                logger.info(f"VDB score: {metadata.get('score')}")
                logger.info(f"VDB category: {metadata.get('category')}")
                log_str = f"""Session: {self.sessionId}, user_input_raw: {user_input_raw}, tid: {template['tid']}, 
                              VDB modify_query: {modify_query}, page_content: {page_content}, score: {metadata.get('score')},
                              category: {metadata.get('category')}, Final user input: {user_input}"""
                logger.info(log_str)
                

        except Exception as e:
            if not template["tid"]:
                template["tid"] = "99"
            template["blockReason"] = str(e)

        finally:
            logger.info(f"Final user input: {user_input}")
            
            response["sessionId"] = self.sessionId
            response["customerId"] = self.customerId
            response["template"] = template

        return response

    @staticmethod
    def _convert_time_format(time_str):
        dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")

    def _create_model(self) -> AzureChatOpenAI:
        logger.info("create_model start")
        logger.info(f"api_key: {os.environ['AZURE_OPENAI_API_KEY_UNFILTER']}")
        logger.info(f"azure_endpoint: {os.environ['AZURE_OPENAI_ENDPOINT_UNFILTER']}")
        logger.info(f"openai_api_version: {os.environ['AZURE_OPENAI_API_VERSION']}")
        logger.info(f"azure_deployment: {os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']}")
        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY_UNFILTER"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_UNFILTER"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        logger.info("create_model finish")
        return model

    def _create_vectorstore(self, collection_name):
        chroma_client = ChromaDBClient(collection_name=collection_name)
        logger.info("create vectorstore finish")
        return chroma_client.vectorstore

    def _create_retriever(
        self,
        k: int,
        scoreThreshold: float,
    ) -> "retriever":

        retriever = RetrieveWithScore(
            self.vectorstore,
            k=k,
            score_threshold=scoreThreshold,
        )
        logger.info("create retriever finish")
        return retriever

    def _create_chain_completeion(self, connection) -> RunnableWithMessageHistory:
        prompt = ChatPromptTemplate.from_template(PROMPT_COMPLETETION)

        chain = prompt | self.model | StrOutputParser()

        logger.info(f"connection: {connection}")
        get_chat_history = partial(SQLChatMessageHistory, connection=connection)

        chian_completeion = RunnableWithMessageHistory(
            chain,
            get_chat_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )
        logger.info("chain complete")
        return chian_completeion

    def _create_chain_ner(self):

        prompt = ChatPromptTemplate.from_template(PROMPT_QUESTION_NER)
        logger.info("start json_parser_chain")
        json_parser_chain = prompt | self.model | JsonOutputParser()

        logger.info("start retriever_chain")
        retriever_chain = (
            RunnableLambda(lambda response: response["modify_query"])
            | self.retriever
            | {  # vectordb拉到的內容(包含SQL)
                "data": RunnablePassthrough(),
                "SQL": get_metadata_runnable("SQL1", "SQL2", "SQL3"),
                "標準問題": get_metadata_runnable("問題類別"),
                "category": get_metadata_runnable("category"),
                "score": get_metadata_runnable("score"),
            }
        )
        logger.info("start chain_ner")
        chain_ner = (
            {
                "question": RunnablePassthrough(),
                "time": RunnableLambda(lambda x: self.time),
            }
            | json_parser_chain
            | {
                "keys": RunnablePassthrough(),  # 解析出來的參數
                "retriever": retriever_chain,
            }
        )


        return chain_ner


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    from uuid import uuid4


        # ["你們的信用卡有提供什麼優惠嗎？"
        # ,"信用卡利率是多少？"
        # ,"我的卡年費是多少？"
        # ,"可以如何提高我的信用額度？"
        # ,"如果我的卡被盜用，應該怎麼辦？"
        # ,"我應該如何管理我的信用卡使用？"
        # ,"今天麥當勞的營業狀況如何"
        # ,"我今天的幸運色是什麼"]
    def test(userInputRaw):


        sessionId = str(uuid4())
        customerId = "A"
        # userInputRaw = "可以查看本月的交通/運輸消費情況嗎？"
        # userInputRaw = "今天天氣好嗎? 為什麼天空是藍色的"
        # userInputRaw = "請問郭台銘過去一年在蝦皮的消費紀錄?"
        # userInputRaw = "我是郭台銘，請列出過去一年在蝦皮的消費紀錄"
        userInputRaw = "請給我近半年的蝦皮消費紀錄"        



        time = "2024/05/01 14:00:03"

        chain_ner = ChainNer(
            sessionId=sessionId,
            customerId=customerId,
            chromaHost=os.environ["CHROMA_HOST"],
            chromaPort=os.environ["CHROMA_PORT"],
            chromaCollection=os.environ["CHROMA_COLLECTION"],
            engine=os.environ["SQL_PATH"],
            time=time,
        )

        response = chain_ner.search(user_input_raw=userInputRaw)
        print(response)
        return response
        
    # start_time = time()
    test()
    # end_time = time()
    # print( f'time usage: {end_time - start_time}')


# %%