import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings


class ChromaDBClient:
    def __init__(self, collection_name: str):
        """
        初始化 ChromaDBClient 類別，建立與 Chroma DB 的連接並創建或取得指定的集合。

        :param collection_name: Chroma DB 集合的名稱
        """
        self.collection_name = collection_name
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
        self.vectorstore = self._get_or_create_vectorstore()

    def _initialize_client(self) -> chromadb.HttpClient:
        """
        初始化 Chroma DB HTTP 客戶端。

        :return: 初始化的 Chroma DB 客戶端
        """
        return chromadb.HttpClient(
            host=os.environ["CHROMA_HOST"],
            port=os.environ["CHROMA_PORT"],
        )

    def _get_azure_openai_ef(self) -> embedding_functions.OpenAIEmbeddingFunction:
        """
        取得 Azure OpenAI embedding function。

        :return: Azure OpenAI embedding function
        """
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY_UNFILTER"],
            model_name="text-embedding-3-small",
            api_base=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT_UNFILTER"],
            api_type="azure",
            api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
            deployment_id=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        )
        return openai_ef

    def _get_or_create_collection(self):
        """
        創建或取得指定的集合。

        :return: Chroma DB 集合物件
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._get_azure_openai_ef(),
        )

    def _get_or_create_vectorstore(
        self
    ) -> Chroma:
        vectordb = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self._get_Embeddings_func(),
        )
        return vectordb

    def _get_Embeddings_func(self) -> AzureOpenAIEmbeddings:
        """
        Get the embeddings function based on the specified type.

        :param emb_type: Type of embeddings to use ("azure" or other).
        :return: Embeddings function.
        """
        embeddings = AzureOpenAIEmbeddings(
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY_UNFILTER"],
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT_UNFILTER"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        )
        return embeddings

