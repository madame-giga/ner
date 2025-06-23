import transformers
from langchain_huggingface import HuggingFaceEndpoint
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from huggingface_hub import login 

login(token='')

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"loading model: {repo_id}", end='')

llm = HuggingFaceEndpoint(
    repo_id = repo_id
)

print("...DONE")
llm_transformer = LLMGraphTransformer(llm=llm)
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
"""

doc = Document(page_content=text)

graph_doc = llm_transformer.convert_to_graph_documents([doc])
print(graph_doc)
