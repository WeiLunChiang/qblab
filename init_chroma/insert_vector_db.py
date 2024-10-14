
import itertools
import os
import pickle
import time
import pandas as pd
import chromadb
from tqdm import tqdm
from typing import List, Dict, Any, Union
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


class CustomCSVLoader(CSVLoader):
    """
    Custom CSV Loader that modifies page content by removing a
    specified prefix.
    """

    def __init__(self, file_path: str, metadata_columns: List[str], source_column: str):
        """
        Initialize with file path, metadata columns, and source
        column.

        :param file_path: Path to the CSV file.
        :param metadata_columns: List of metadata column names.
        :param source_column: Name of the source column.
        """
        super().__init__(
            file_path=file_path,
            metadata_columns=metadata_columns,
            source_column=source_column,
        )
        self.file_name = os.path.basename(file_path)
        self.source_column = source_column

    def load(self) -> List[Document]:
        """
        Load and process CSV file, removing prefix from page content.

        :return: List of processed Document objects.
        """
        docs = super().load()
        prefix = f"{self.source_column}: "
        for doc in docs:
            if doc.page_content.startswith(prefix):
                doc.page_content = doc.page_content[len(prefix) :]
            doc.metadata["source"] = self.file_name
        return docs


def delete_collection(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
) -> None:
    client = chromadb.HttpClient(
        host=host,
        port=port,
    )
    collection = client.get_or_create_collection(name=collection_name)
    client.delete_collection(name=collection_name)
    print(f"delete collection: {collection_name}")


def _get_Embeddings_func(
    emb_type: str = "azure",
) -> Union[AzureOpenAIEmbeddings, OpenAIEmbeddings]:
    """
    Get the embeddings function based on the specified type.

    :param emb_type: Type of embeddings to use ("azure" or other).
    :return: Embeddings function.
    """
    if emb_type == "azure":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


def get_or_create_http_chromadb(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
    emb_type: str = "azure",
) -> Chroma:
    """
    Get or create a ChromaDB instance via HTTP with the specified
    host, port, and collection name.

    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :param emb_type: Type of embeddings to use ("azure" or other).
    :return: ChromaDB instance.
    """
    client = chromadb.HttpClient(
        host=host,
        port=port,
    )
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_get_Embeddings_func(emb_type=emb_type),
    )
    return vectordb


def insert_to_vector_db(
    num_batches=1,
    max_retries=20,
    file_path="qa_set_with_sql_lite.csv",
    host="127.0.0.1",
    port="8000",
    collection_name="collect_cubelab_qa_lite",
    metadata_columns=["category", "變形問題", "SQL1", "SQL2", "SQL3"],
    source_column="問題類別",
):
    # 載入CSV檔案，並指定metadata欄位
    loader = CustomCSVLoader(
        file_path=file_path,
        metadata_columns=metadata_columns,
        source_column=source_column,
    )
    docs = loader.load()
    # breakpoint()
    list_length = len(docs)
    num_batches = num_batches
    batch_size = list_length // num_batches
    vector_db = get_or_create_http_chromadb(
        host=host,
        port=port,
        collection_name=collection_name,
        emb_type="azure",
    )

    max_retries = max_retries
    resume_file = "resume.pkl"

    # 如果resume_file存在，則讀取c的值，否則將c設置為初始值0
    if os.path.exists(resume_file):
        with open(resume_file, "rb") as f:
            c = pickle.load(f)
    else:
        c = 0

    # 進行資料批次上傳
    for _ in range(max_retries):
        try:
            for i in tqdm(range(c, num_batches)):
                c = i
                batch = list(
                    itertools.islice(docs, i * batch_size, (i + 1) * batch_size)
                )
                # breakpoint()
                vector_db.add_documents(batch)
                # 每次成功處理一個batch後，將c的值寫入resume_file
                with open(resume_file, "wb") as f:
                    pickle.dump(c, f)
                break
            # 完成所有批次上傳後，刪除resume_file
            if os.path.exists(resume_file):
                os.remove(resume_file)
            break  # 如果操作成功，則跳出迴圈
        except Exception as e:
            print(f"錯誤: {e}, 正在重試 ({_+1}/{max_retries})")
            time.sleep(60)
    else:
        print("已達到最大重試次數，操作失敗")


def create_csv_ner(file_path="cubelab.xlsx", output_path="qa_set_with_sql_lite.csv"):
    qa = pd.read_excel(file_path, sheet_name="QA")
    qa.rename(
        {
            "編號": "category",
            "Q": "變形問題",
        },
        axis=1,
        inplace=True,
    )
    qa.to_csv(output_path, index=False)
    print(qa.columns)


if __name__ == "__main__":
    input_path = "cubelab.xlsx"
    data_path = "qa_set_with_sql_lite.csv"
    collection_name = "collect_cubelab_qa_lite"

    create_csv_ner(input_path, data_path)
    print(os.environ["CHROMA_HOST"])

    delete_collection(
        host=os.environ["CHROMA_HOST"],  # "127.0.0.1"
        port=os.environ["CHROMA_PORT"],  # "8000"
        collection_name=collection_name,
    )

    insert_to_vector_db(
        file_path=data_path,
        host=os.environ["CHROMA_HOST"],  # "127.0.0.1"
        port=os.environ["CHROMA_PORT"],  # "8000"
        collection_name=collection_name,
        metadata_columns=["category", "問題類別", "SQL1", "SQL2", "SQL3"],
        source_column="變形問題",
    )
