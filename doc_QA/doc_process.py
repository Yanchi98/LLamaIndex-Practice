import logging
import os
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

project_dir = os.path.dirname(os.path.abspath(__file__))

# 日志打印
logger = logging.getLogger()
logger.handlers.clear()  # 清除所有已存在的 handlers
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Settings.embed_model = HuggingFaceEmbedding(
    model_name = f"{project_dir}/model/Xorbits/bge-m3/"
)

logger.info("加载embedding模型成功")

Settings.transformations = [SentenceSplitter(chunk_size=256)]
logger.info("设置切片策略成功")

documents = SimpleDirectoryReader(f"{project_dir}/data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)
logger.info("离线数据加载成功")

index.storage_context.persist()
logger.info("离线数据存储到storage目录中")

