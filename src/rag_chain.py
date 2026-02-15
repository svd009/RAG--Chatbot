from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def create_rag_chain(vectorstore):
    """Production RAG chain using LangChain create_retrieval_chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # RAG prompt template
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer "
        "the question. If you don't know the answer, say you don't know. "
        "Keep answers concise and accurate.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Production RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 4}), 
        question_answer_chain
    )
    
    return retrieval_chain
