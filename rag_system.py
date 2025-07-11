import os
os.environ["HUGGINGFACE_API_KEY"] = "Your_key"
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from typing import List
from langchain_huggingface import HuggingFaceEndpoint

# 무료 모델들
FREE_MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base", 
    "microsoft-dialo": "microsoft/DialoGPT-medium",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2"
}
selected_model = FREE_MODELS["flan-t5-small"]

# 의약품 관련 URL들
urls = [
    "https://nedrug.mfds.go.kr/index",
    "https://www.health.kr/",
    "https://www.kdra.or.kr/website/index.php",
    "https://www.druginfo.co.kr/",
    "https://www.health.kr/",
    "https://www.mfds.go.kr/",
]

# 웹 문서 로드
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 무료 Hugging Face 임베딩 모델 사용
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 벡터 데이터베이스 생성
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./chroma_db",
)
vectorstore.persist()

# 최신 권장 LLM 엔드포인트 사용
llm = HuggingFaceEndpoint(
    repo_id=selected_model,
    task="text2text-generation",
    model_kwargs={
        "temperature": 0.7,
        "max_length": 512,
        "do_sample": True,
        "top_p": 0.9
    }
)

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    search_results: List[str]

def search_node(state: AgentState) -> AgentState:
    global vectorstore
    query = state["query"]
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(query)  # 최신 방식
    context = [doc.page_content for doc in docs]
    return {
        "query": query,
        "context": context,
        "search_results": context,
        "answer": ""
    }

def answer_node(state: AgentState) -> AgentState:
    global llm
    query = state["query"]
    context = state["context"]
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        다음 컨텍스트를 기반으로 질문에 답변해주세요.
        
        컨텍스트:
        {context}
        
        질문: {question}
        
        답변:"""
    )
    combined_context = "\n\n".join(context)
    response = llm.invoke(
        prompt_template.format(
            context=combined_context,
            question=query
        )
    )
    return {
        "query": query,
        "context": context,
        "search_results": state["search_results"],
        "answer": response
    }

def create_rag_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    app = workflow.compile()
    return app

def query_rag_agent(query: str):
    agent = create_rag_agent()
    initial_state: AgentState = {
        "query": query,
        "context": [],
        "answer": "",
        "search_results": []
    }
    result = agent.invoke(initial_state)
    return result

def change_model(model_name: str):
    global llm, selected_model
    if model_name in FREE_MODELS:
        selected_model = FREE_MODELS[model_name]
        llm = HuggingFaceEndpoint(
            repo_id=selected_model,
            task="text2text-generation",
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "do_sample": True,
                "top_p": 0.9
            }
        )
        print(f"모델이 {model_name}로 변경되었습니다.")
    else:
        print(f"사용 가능한 모델: {list(FREE_MODELS.keys())}")

if __name__ == "__main__":
    test_query = "의약품 안전성에 대해 알려주세요"
    result = query_rag_agent(test_query)
    print(f"질문: {test_query}")
    print(f"답변: {result['answer']}")
    print(f"참고 문서 수: {len(result['context'])}")
    print(f"사용 중인 모델: {selected_model}") 