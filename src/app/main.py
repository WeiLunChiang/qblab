import logging
import os
import orjson
import uvicorn
import mlflow
from fastapi import FastAPI, Request
from app.schemas.response import (
    ChatResponse, GenaiResponse, EvaluateResponse, MWHeader, ChatTranRS,
    GenAIResponseTranRS, EvaluateTranRS, GenAIModel, BaseGenaiTemplate, ChatTemplate
)
from app.setting.utils_mlflow import mlflow_exception_logger, mlflow_openai_callback
from app.setting.config import Settings
from app.gai_executors.chain_ner import ChainNer
from app.gai_executors.gai_response import GenAIResponse
from app.db.conn import sqlalchemy_engine
from app.db.model import Base
from app.db.insert_db import insert_evaluation
from app.setting.exceptions import *


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "30"
tracking_server_uri = os.environ["TRACKING_SERVER_URI"]
mlflow.set_tracking_uri(tracking_server_uri)
mlflow.set_experiment("cubelab_demo")
mlflow.langchain.autolog()

app = FastAPI()


# 定义路由
@app.post("/api/card-consumption/chat")
# @mlflow_exception_logger
@mlflow_openai_callback
async def chat(request: Request):
    jdata = await request.json()
    logger.info(f"request: {jdata}")
    collection_name = Settings.VDB_COLLECTION
    logger.info(f"collection_name: {collection_name}")
    response = ChatResponse()
    mwheader = MWHeader(MSGID=jdata["MWHEADER"]["MSGID"],
                        SOURCECHANNEL=jdata["MWHEADER"]["SOURCECHANNEL"],
                        TXNSEQ=jdata["MWHEADER"]["TXNSEQ"]
    )

    try:
        logger.info(f"start chain ner")
        chain_ner = ChainNer(
            sessionId=jdata["TRANRQ"]["sessionId"],
            customerId=jdata["TRANRQ"]["customerId"],
            chromaCollection=collection_name,
            engine=sqlalchemy_engine,
            time=jdata["TRANRQ"]["time"]
        )
        logger.info("Handling chat request")
        logger.info(f"message: {jdata['TRANRQ']['message']}")
        gai_response = chain_ner.search(user_input=jdata["TRANRQ"]["message"])
        logger.info(f"GAI Response: {gai_response}")

        # 建立 ChatTranRS 回應
        tranrs = ChatTranRS(**gai_response)

        # 建立 ChatResponse 回應
        response = ChatResponse(
            MWHEADER=mwheader,
            TRANRS=tranrs
        )
        
    except CathayDefinedException as e:
        response.MWHEADER.RETURNCODE = e.error_code
        response.MWHEADER.RETURNDESC = e.error_describe
        logger.exception(e, extra=response.MWHEADER.dict())
    except Exception as e:
        response.MWHEADER.RETURNCODE = "9999"
        response.MWHEADER.RETURNDESC = "其他異常錯誤"
        logger.exception(e, extra=response.MWHEADER.dict())
    else:
        logger.info("Processed successfully", extra=response.MWHEADER.dict())
    finally:
        logger.info("Return Response", extra=response.dict())
        return response.dict()

@app.post("/api/card-consumption/genai-response")
# @mlflow_exception_logger
@mlflow_openai_callback
async def gai_response(request: Request):
    jdata = await request.json()
    logger.info(f"request: {jdata}")
    response = GenaiResponse()
    mwheader = MWHeader(MSGID=jdata["MWHEADER"]["MSGID"],
                        SOURCECHANNEL=jdata["MWHEADER"]["SOURCECHANNEL"],
                        TXNSEQ=jdata["MWHEADER"]["TXNSEQ"]
    )

    try:
        genai_response = GenAIResponse(
            sessionId=jdata["TRANRQ"]["sessionId"],
            customerId=jdata["TRANRQ"]["customerId"]
        )

        gai_response_msg = genai_response.generate_answer(
            message=jdata["TRANRQ"]["message"],
            tid=jdata["TRANRQ"]["tid"],
            consumptionNumber=jdata["TRANRQ"]["consumptionNumber"],
            totalAmount=jdata["TRANRQ"]["totalAmount"],
            storeName=jdata["TRANRQ"]["storeName"],
            categoryName=jdata["TRANRQ"]["categoryName"]
        )
        logger.info(f"GAI response msg: {gai_response_msg}")
        
        tranrs = GenAIResponseTranRS(**gai_response_msg)
        # 建立 ChatResponse 回應
        response = GenaiResponse(
            MWHEADER=mwheader,
            TRANRS=tranrs
        )

    except CathayDefinedException as e:
        response.MWHEADER.RETURNCODE = e.error_code
        response.MWHEADER.RETURNDESC = e.error_describe
        logger.exception(e, extra=response.MWHEADER.dict())
    except Exception as e:
        response.MWHEADER.RETURNCODE = "9999"
        response.MWHEADER.RETURNDESC = "其他異常錯誤"
        logger.exception(e, extra=response.MWHEADER.dict())
    else:
        logger.info("Processed successfully", extra=response.MWHEADER.dict())
    finally:
        logger.info("Return Response", extra=response.dict())
        return response.dict()

@app.post("/api/card-consumption/evaluate")
async def evaluate(request: Request):
    jdata = await request.json()
    logger.info(f"request: {jdata}")
    response = EvaluateResponse()
    response.MWHEADER.MSGID = jdata["MWHEADER"]["MSGID"]
    response.MWHEADER.SOURCECHANNEL = jdata["MWHEADER"]["SOURCECHANNEL"]
    response.MWHEADER.TXNSEQ = jdata["MWHEADER"]["TXNSEQ"]

    response.TRANRS.sessionId = jdata["TRANRQ"]["sessionId"]
    response.TRANRS.customerId = jdata["TRANRQ"]["customerId"]

    try:
        evaluation_data = {
            "TXNSEQ": jdata["MWHEADER"]["TXNSEQ"],
            "SESSIONI_ID": jdata["TRANRQ"]["sessionId"],
            "CUSTOMER_ID": jdata["TRANRQ"]["customerId"],
            "EVALUATE": jdata["TRANRQ"]["evaluate"],
            "TIME": jdata["TRANRQ"]["time"]
        }
        insert_evaluation(obj=evaluation_data,
                          con=sqlalchemy_engine)

    except CathayDefinedException as e:
        response.MWHEADER.RETURNCODE = e.error_code
        response.MWHEADER.RETURNDESC = e.error_describe
        logger.exception(e, extra=response.MWHEADER.dict())
    except Exception as e:
        response.MWHEADER.RETURNCODE = "9999"
        response.MWHEADER.RETURNDESC = "其他異常錯誤"
        logger.exception(e, extra=response.MWHEADER.dict())
    else:
        logger.info("Processed successfully", extra=response.MWHEADER.dict())
    finally:
        logger.info("Return Response", extra=response.dict())
        return response.dict()


if __name__ == "__main__":    
    Base.metadata.create_all(bind=sqlalchemy_engine)
    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=True)