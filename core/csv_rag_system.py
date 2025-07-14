import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 무료 모델들
FREE_MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base", 
    "microsoft-dialo": "microsoft/DialoGPT-medium",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2"
}
selected_model = FREE_MODELS["flan-t5-small"]

class CSVMedicineRAG:
    def __init__(self, csv_file_path: str, persist_directory: str = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/ddd/RAG-system-with-Huggingface/chroma_db_new"):
        """
        CSV 파일을 사용한 의약품 RAG 시스템 초기화
        
        Args:
            csv_file_path: CSV 파일 경로
            persist_directory: 벡터 데이터베이스 저장 디렉토리
        """
        self.csv_file_path = csv_file_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        
        # CSV 컬럼명 정의
        self.columns = [
            "수입제조국", "마약구분", "ATC코드", "제조/수입", "모양", "품목분류",
            "업체명", "제품명", "단축", "주성분영문", "묶음의약품정보", "품목구분",
            "허가번호", "제품영문명", "제형", "취소/취하일자", "표준코드명", "첨가제",
            "e은약요", "허가일", "신약구분", "허가/신고", "취소/취하", "업체영문명",
            "장축", "색상", "전문의약품", "품목기준코드", "완제/원료", "주성분"
        ]
        
        self._initialize_components()
    
    def _initialize_components(self):
        """LLM과 임베딩 모델 초기화"""
        try:
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # LLM 초기화
            self.llm = HuggingFaceEndpoint(
                model=selected_model,
                task="text2text-generation",
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                max_new_tokens=512
            )
            
            logger.info("컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def load_csv_data(self):
        """CSV 파일을 로드하고 데이터를 전처리"""
        try:
            logger.info(f"CSV 파일 로딩 중: {self.csv_file_path}")
            
            # CSV 파일 읽기
            df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            
            # 컬럼명 확인 및 수정
            if len(df.columns) != len(self.columns):
                logger.warning(f"예상 컬럼 수: {len(self.columns)}, 실제 컬럼 수: {len(df.columns)}")
                logger.info(f"실제 컬럼명: {list(df.columns)}")
            
            # NaN 값 처리
            df = df.fillna("")
            
            logger.info(f"데이터 로딩 완료: {len(df)} 행")
            return df
            
        except Exception as e:
            logger.error(f"CSV 파일 로딩 실패: {e}")
            raise
    
    def create_documents_from_csv(self, df):
        """DataFrame을 Document 객체로 변환"""
        documents = []
        
        for idx, row in df.iterrows():
            # 각 행의 모든 정보를 하나의 텍스트로 결합
            content_parts = []
            
            for col in df.columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    content_parts.append(f"{col}: {row[col]}")
            
            if content_parts:
                content = "\n".join(content_parts)
                
                # 메타데이터 생성
                metadata = {
                    "row_index": idx,
                    "product_name": row.get("제품명", ""),
                    "company_name": row.get("업체명", ""),
                    "license_number": row.get("허가번호", ""),
                    "main_ingredient": row.get("주성분", ""),
                    "dosage_form": row.get("제형", ""),
                    "classification": row.get("품목분류", "")
                }
                
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(document)
        
        logger.info(f"문서 생성 완료: {len(documents)} 개")
        return documents
    
    def build_vectorstore(self):
        """벡터 데이터베이스 구축"""
        try:
            # CSV 데이터 로드
            df = self.load_csv_data()
            logger.info(f"CSV 데이터 로드 완료: {len(df)} 행")
            
            # Document 객체 생성
            documents = self.create_documents_from_csv(df)
            logger.info(f"문서 생성 완료: {len(documents)} 개")
            
            if not documents:
                raise ValueError("생성된 문서가 없습니다. CSV 파일을 확인해주세요.")
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500,  # 의약품 정보는 더 큰 청크로 설정
                chunk_overlap=50
            )
            doc_splits = text_splitter.split_documents(documents)
            
            logger.info(f"텍스트 분할 완료: {len(doc_splits)} 개 청크")
            
            if not doc_splits:
                raise ValueError("분할된 문서가 없습니다.")
            
            # 벡터 데이터베이스 생성
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="medicine-rag-chroma",
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            
            logger.info("벡터 데이터베이스 구축 완료")
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스 구축 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise
    
    def load_existing_vectorstore(self):
        """기존 벡터 데이터베이스 로드"""
        try:
            self.vectorstore = Chroma(
                collection_name="medicine-rag-chroma",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("기존 벡터 데이터베이스 로드 완료")
            return True
        except Exception as e:
            logger.warning(f"기존 벡터 데이터베이스 로드 실패: {e}")
            return False

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    search_results: List[str]

def create_medicine_rag_agent(csv_rag: CSVMedicineRAG):
    """의약품 RAG 에이전트 생성"""
    
    def search_node(state: AgentState) -> AgentState:
        try:
            query = state["query"]
            logger.info(f"검색 쿼리: {query}")
            
            if csv_rag.vectorstore is None:
                raise ValueError("벡터스토어가 초기화되지 않았습니다.")
            
            retriever = csv_rag.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            docs = retriever.invoke(query)
            context = [doc.page_content for doc in docs]
            
            logger.info(f"검색된 문서 수: {len(docs)}")
            if docs:
                logger.info(f"첫 번째 문서 샘플: {docs[0].page_content[:200]}...")
            
            return {
                "query": query,
                "context": context,
                "search_results": context,
                "answer": ""
            }
        except Exception as e:
            logger.error(f"검색 노드 오류: {e}")
            import traceback
            logger.error(f"검색 상세 오류: {traceback.format_exc()}")
            return {
                "query": query,
                "context": [],
                "search_results": [],
                "answer": f"[검색 오류] {e}"
            }

    def answer_node(state: AgentState) -> AgentState:
        try:
            query = state["query"]
            context = state["context"]
            
            if csv_rag.llm is None:
                raise ValueError("LLM이 초기화되지 않았습니다.")
            
            # 컨텍스트가 비어있는 경우 처리
            if not context:
                return {
                    "query": query,
                    "context": context,
                    "search_results": state["search_results"],
                    "answer": "죄송합니다. 질문과 관련된 의약품 정보를 찾을 수 없습니다. 다른 키워드로 다시 질문해주세요."
                }
            
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                다음 의약품 정보를 기반으로 질문에 답변해주세요.
                의약품 정보는 정확하고 신뢰할 수 있는 정보를 제공해야 합니다.
                
                의약품 정보:
                {context}
                
                질문: {question}
                
                답변:"""
            )
            
            combined_context = "\n\n".join(context)
            logger.info(f"컨텍스트 길이: {len(combined_context)}")
            logger.info(f"컨텍스트 샘플: {combined_context[:200]}...")
            
            response = csv_rag.llm.invoke(
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
        except Exception as e:
            logger.error(f"답변 노드 오류: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                "query": state.get("query", ""),
                "context": state.get("context", []),
                "search_results": state.get("search_results", []),
                "answer": f"[답변 오류] {e}"
            }

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()

def query_medicine_rag(query: str, csv_rag: CSVMedicineRAG):
    """의약품 RAG 시스템에 질문"""
    agent = create_medicine_rag_agent(csv_rag)
    initial_state: AgentState = {
        "query": query,
        "context": [],
        "answer": "",
        "search_results": []
    }
    result = agent.invoke(initial_state)
    return result

def change_model(model_name: str, csv_rag: CSVMedicineRAG):
    """모델 변경"""
    global selected_model
    if model_name in FREE_MODELS:
        selected_model = FREE_MODELS[model_name]
        csv_rag.llm = HuggingFaceEndpoint(
            model=selected_model,
            task="text2text-generation",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=512
        )
        logger.info(f"모델이 {model_name}로 변경되었습니다.")
    else:
        logger.warning(f"사용 가능한 모델: {list(FREE_MODELS.keys())}")

if __name__ == "__main__":
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    try:
        # CSV RAG 시스템 초기화
        csv_rag = CSVMedicineRAG(csv_file_path)
        
        # 기존 벡터 데이터베이스 삭제 후 새로 구축
        import shutil
        if os.path.exists(csv_rag.persist_directory):
            logger.info("기존 벡터 데이터베이스 삭제 중...")
            shutil.rmtree(csv_rag.persist_directory)
        
        logger.info("새로운 벡터 데이터베이스 구축 시작")
        csv_rag.build_vectorstore()
        logger.info("벡터 데이터베이스 구축 완료!")
        
        # 테스트 질문
        test_queries = [
            "엑스원에이정에 대한 정보를 알려주세요",
            "보건용마스크 중에서 어떤 것들이 있나요?",
            "파비안디정에 대해서 알려주세요",
            "특정 업체의 제품을 찾고 싶어요"
        ]
        
        for query in test_queries:
            logger.info(f"\n질문: {query}")
            result = query_medicine_rag(query, csv_rag)
            logger.info(f"답변: {result['answer']}")
            logger.info(f"참고 문서 수: {len(result['context'])}")
            
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}") 