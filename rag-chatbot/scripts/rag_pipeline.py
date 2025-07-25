from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0},
        task="text2text-generation",
        huggingfacehub_api_token="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"
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

def ask_question(chain, question):
    return chain.run(question)
