import os
import logging
import torch
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from transformers import BitsAndBytesConfig

project_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = f"{project_dir}/storage"

# 日志打印
logger = logging.getLogger()
logger.handlers.clear()  # 清除所有已存在的 handlers
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger.info(f"项目地址：{project_dir}")

def completion_to_prompt(completion):
   return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

Settings.llm = HuggingFaceLLM(
    model_name=f"{project_dir}/model/qwen",
    tokenizer_name=f"{project_dir}/model/qwen",
    #model_name="Qwen/Qwen2-7B-Instruct",
    #tokenizer_name="Qwen/Qwen2-7B-Instruct",
    context_window=5000,
    max_new_tokens=2000,
    generate_kwargs={"temperature": 0.1, "top_k": 5, "top_p": 0.1},
    model_kwargs={"quantization_config": quantization_config},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name = f"{project_dir}/model/Xorbits/bge-m3/"
)

logger.info("加载模型成功")

storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
    vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir=persist_dir
    ),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
)
logger.info("加载数据库成功")

index = load_index_from_storage(storage_context)
    
def get_query_engine():
    return index.as_query_engine(similarity_top_k=5)
# your_query = "楚子航和夏弥是什么关系？"
# response = query_engine.query(your_query)
# print(f"Answer: {response.response}")

# for id, item in enumerate(response.source_nodes):
#     node = item.node
#     print(f'[{id+1}]' + node.text)

