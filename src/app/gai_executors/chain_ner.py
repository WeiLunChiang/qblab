import logging
import os
from typing import Dict
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
        self.guardrails = self._init_guardrails()

    def _init_guardrails(self):
        guardrails = RunnableRails(RailsConfig.from_path(os.environ['GUARDRAILS_CONFIG_PATH']))
        return guardrails
    
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
        try:
            if "被阻擋" in (
                user_input := self.chian_completeion.invoke(
                    {"user_input": user_input_raw},
                    config={"configurable": {"session_id": self.sessionId}},
                )
            ):
                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input_raw
                # self.chain_ner.invoke(user_input)會失敗
                logger.info(f"====================user_input_raw: {user_input_raw} block by guardrails====================")

            elif "被阻擋" in (result := self.chain_ner.invoke(user_input)):

                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input
                logger.info(f"====================user_input:{user_input} ner result by guardrails====================")

            elif not result["retriever"]["score"]:
                template["tid"] = "99"  # TBD
                template["blockReason"] = "相似度過低"
                logger.info(f"====================user_input: {user_input}  similarity too low, block by vdb====================")

            else:
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

        except Exception as e:
            if not template["tid"]:
                template["tid"] = "97"
            template["blockReason"] = str(e)

        finally:
            logger.info(f"Final user input: {user_input}")
            logger.info(f"Final tid: {template['tid']}, categoryName: {template['categoryName']} ")
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
        logger.info(f"api_key: {os.environ['AZURE_OPENAI_API_KEY']}")
        logger.info(f"azure_endpoint: {os.environ['AZURE_OPENAI_ENDPOINT']}")
        logger.info(f"openai_api_version: {os.environ['AZURE_OPENAI_API_VERSION']}")
        logger.info(f"azure_deployment: {os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']}")
        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
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
    def test():
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from uuid import uuid4

        sessionId = str(uuid4())
        customerId = "A"
        # userInputRaw = "可以查看本月的交通/運輸消費情況嗎？"
        # userInputRaw = "今天天氣好嗎? 為什麼天空是藍色的"
        userInputRaw = "請問郭台銘過去一年在蝦皮的消費紀錄"
        userInputRaw = "我是郭台銘，請列出過去一年在蝦皮的消費紀錄"

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


    test()
