#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 CSV 기반 의약품 RAG 시스템
- 키워드 추출 및 가중치 검색
- 개선된 청크 생성
- 한국어 특화 검색
"""

import os
import pandas as pd
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

class ImprovedCSVMedicineRAG:
    def __init__(self, csv_file_path: str, persist_directory: str = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/ddd/RAG-system-with-Huggingface/chroma_db_improved_v2"):
        """
        향상된 CSV 파일을 사용한 의약품 RAG 시스템 초기화
        """
        self.csv_file_path = csv_file_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # 중요 컬럼들 (검색 우선순위)
        self.important_columns = [
            "제품명", "업체명", "주성분", "품목분류", "제형", "표준코드명"
        ]
        
        # 전체 컬럼명
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
    
    def extract_keywords(self, query: str) -> List[str]:
        """질문에서 키워드 추출 - LLM 활용 + 후처리"""
        try:
            # LLM을 사용한 키워드 추출 프롬프트 (예시 포함)
            keyword_prompt = f"""
            다음 질문에서 의약품명, 성분명, 증상명 등 검색에 필요한 핵심 키워드만 추출해서 쉼표로 구분해 주세요.
            조사, 접미사, 불필요한 단어(예: '에', '에 대해서', '알려줘', '알려주세요', '정보', '어떤', '것들', '있나요', '대해서', '알려주')는 모두 빼고, 순수 키워드만 남기세요.
            예시1: 질문: 아스피린에 대해서 알려줘 → 키워드: 아스피린
            예시2: 질문: 두통약 중에서 어떤 것들이 있나요? → 키워드: 두통약
            예시3: 질문: 타이레놀의 주성분이 뭐야? → 키워드: 타이레놀, 주성분
            
            질문: {query}
            
            키워드:
            """
            if self.llm is not None:
                try:
                    llm_response = self.llm.invoke(keyword_prompt)
                    # 쉼표, 줄바꿈, 공백 기준 분리
                    keywords = [kw.strip() for kw in re.split(r'[\n,]+', llm_response) if kw.strip()]
                    # 불필요한 단어 제거
                    stop_words = ['에', '의', '를', '을', '가', '이', '는', '은', '도', '만', '대한', '정보', '알려줘', '알려주세요', '어떤', '것들', '있나요', '대해서', '알려주']
                    keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 1]
                    logger.info(f"LLM 추출 키워드: {keywords}")
                    return keywords
                except Exception as e:
                    logger.warning(f"LLM 키워드 추출 실패: {e}")
            # LLM 실패 시 fallback
            return self._extract_keywords_fallback(query)
        except Exception as e:
            logger.error(f"키워드 추출 오류: {e}")
            return self._extract_keywords_fallback(query)
    
    def _extract_keywords_fallback(self, query: str) -> List[str]:
        """기존 방식의 키워드 추출 (fallback)"""
        keywords = []
        
        # 1. 한국어 의약품 관련 키워드 패턴
        medicine_patterns = [
            r'[가-힣]+정',  # ~정 (정제)
            r'[가-힣]+액',  # ~액 (액제)
            r'[가-힣]+캡슐',  # ~캡슐
            r'[가-힣]+시럽',  # ~시럽
            r'[가-힣]+주',  # ~주 (주사제)
            r'[가-힣]+크림',  # ~크림
            r'[가-힣]+연고',  # ~연고
            r'[가-힣]+제',  # ~제
            r'[가-힣]+약',  # ~약
            r'[가-힣]+마스크',  # ~마스크
            r'[가-힣]+소독',  # ~소독
            r'[가-힣]+치료제',  # ~치료제
            r'[가-힣]+항생제',  # ~항생제
            r'[가-힣]+소화제',  # ~소화제
            r'[가-힣]+진통제',  # ~진통제
            r'[가-힣]+혈압약',  # ~혈압약
            r'[가-힣]+당뇨약',  # ~당뇨약
        ]
        
        # 패턴 매칭으로 키워드 추출
        for pattern in medicine_patterns:
            matches = re.findall(pattern, query)
            keywords.extend(matches)
        
        # 2. 일반적인 의약품명 추출 (2글자 이상의 연속된 한글)
        medicine_names = re.findall(r'[가-힣]{2,}', query)
        keywords.extend(medicine_names)
        
        # 3. 영어 의약품명 추출
        english_names = re.findall(r'[a-zA-Z]{2,}', query)
        keywords.extend(english_names)
        
        # 4. 조사, 접미사 등 제거
        stop_words = [
            '에', '의', '를', '을', '가', '이', '는', '은', '도', '만', '대한', 
            '정보를', '알려주세요', '어떤', '것들이', '있나요', '대해서', '알려주',
            '뭔가요', '무엇인가요', '알려주세요', '알려줘', '정보', '대해서',
            '에', '이', '가', '을', '를', '의', '는', '은', '도', '만'
        ]
        
        # 조사가 붙은 단어들에서 조사 제거
        cleaned_keywords = []
        for keyword in keywords:
            # 조사 제거
            for stop_word in stop_words:
                if keyword.endswith(stop_word):
                    keyword = keyword[:-len(stop_word)]
                    break
            
            # 2글자 이상이고 stop_words에 없는 경우만 추가
            if len(keyword) >= 2 and keyword not in stop_words:
                cleaned_keywords.append(keyword)
        
        # 중복 제거
        cleaned_keywords = list(set(cleaned_keywords))
        
        logger.info(f"Fallback 추출된 키워드: {cleaned_keywords}")
        return cleaned_keywords
    
    def load_csv_data(self):
        """CSV 파일을 로드하고 데이터를 전처리"""
        try:
            logger.info(f"CSV 파일 로딩 중: {self.csv_file_path}")
            
            # CSV 파일 읽기
            self.df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            
            # 컬럼명 확인 및 수정
            if len(self.df.columns) != len(self.columns):
                logger.warning(f"예상 컬럼 수: {len(self.columns)}, 실제 컬럼 수: {len(self.df.columns)}")
                logger.info(f"실제 컬럼명: {list(self.df.columns)}")
            
            # NaN 값 처리
            self.df = self.df.fillna("")
            
            # TF-IDF 벡터라이저 초기화
            self._initialize_tfidf()
            
            logger.info(f"데이터 로딩 완료: {len(self.df)} 행")
            return self.df
            
        except Exception as e:
            logger.error(f"CSV 파일 로딩 실패: {e}")
            raise
    
    def _initialize_tfidf(self):
        """TF-IDF 벡터라이저 초기화"""
        try:
            # 중요 컬럼들의 텍스트를 결합  
            text_data = []
            for idx, row in self.df.iterrows():
                important_text = " ".join([
                    str(row.get(col, "")) for col in self.important_columns 
                    if col in self.df.columns and pd.notna(row.get(col, ""))
                ])
                text_data.append(important_text)
            
            # TF-IDF 벡터라이저 생성
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=None
            )
            
            # TF-IDF 매트릭스 생성
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
            
            logger.info("TF-IDF 벡터라이저 초기화 완료")
            
        except Exception as e:
            logger.error(f"TF-IDF 초기화 실패: {e}")
    
    def create_improved_documents_from_csv(self, df):
        """개선된 방식으로 DataFrame을 Document 객체로 변환"""
        documents = []
        
        for idx, row in df.iterrows():
            # 중요 정보를 우선적으로 배치
            important_content = []
            other_content = []
            
            for col in df.columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    content = f"{col}: {row[col]}"
                    
                    if col in self.important_columns:
                        important_content.append(content)
                    else:
                        other_content.append(content)
            
            # 중요 정보를 먼저 배치하고, 나머지 정보를 추가
            if important_content:
                content = "\n".join(important_content)
                if other_content:
                    content += "\n\n" + "\n".join(other_content)
            else:
                content = "\n".join(other_content)
            
            if content.strip():
                # 메타데이터 생성
                metadata = {
                    "row_index": idx,
                    "product_name": row.get("제품명", ""),
                    "company_name": row.get("업체명", ""),
                    "license_number": row.get("허가번호", ""),
                    "main_ingredient": row.get("주성분", ""),
                    "dosage_form": row.get("제형", ""),
                    "classification": row.get("품목분류", ""),
                    "important_text": " ".join([
                        str(row.get(col, "")) for col in self.important_columns 
                        if col in df.columns and pd.notna(row.get(col, ""))
                    ])
                }
                
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(document)
        
        logger.info(f"개선된 문서 생성 완료: {len(documents)} 개")
        return documents
    
    def weighted_search(self, query: str, keywords: List[str], k: int = 5):
        """가중치를 적용한 검색"""
        try:
            # 벡터스토어 상태 확인
            if self.vectorstore is None:
                logger.error("벡터스토어가 None입니다.")
                return []
            
            if not hasattr(self.vectorstore, 'as_retriever'):
                logger.error("벡터스토어에 as_retriever 메서드가 없습니다.")
                return []
            
            # 1. 벡터 검색
            vector_results = []
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k * 2}  # 더 많은 결과를 가져와서 필터링
                )
                vector_docs = retriever.invoke(query)
                vector_results = [(doc, 0.6) for doc in vector_docs]  # 벡터 검색 가중치 0.6
                logger.info(f"벡터 검색 결과: {len(vector_docs)}개")
            except Exception as e:
                logger.error(f"벡터 검색 오류: {e}")
                vector_results = []
            
            # 2. 키워드 기반 검색
            keyword_results = []
            if keywords and self.tfidf_vectorizer is not None:
                # 키워드로 TF-IDF 검색
                keyword_query = " ".join(keywords)
                query_vector = self.tfidf_vectorizer.transform([keyword_query])
                
                # 코사인 유사도 계산
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                
                # 상위 결과 선택
                top_indices = similarities.argsort()[-k*2:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # 임계값 설정
                        # 해당 행의 데이터로 Document 생성
                        row = self.df.iloc[idx]
                        content = self._create_document_content(row)
                        metadata = {
                            "row_index": idx,
                            "product_name": row.get("제품명", ""),
                            "company_name": row.get("업체명", ""),
                            "main_ingredient": row.get("주성분", ""),
                            "similarity_score": float(similarities[idx])
                        }
                        
                        doc = Document(page_content=content, metadata=metadata)
                        keyword_results.append((doc, 0.8))  # 키워드 검색 가중치 0.8
            
            # 3. 정확한 매칭 검색
            exact_results = []
            for keyword in keywords:
                for idx, row in self.df.iterrows():
                    for col in self.important_columns:
                        if col in self.df.columns:
                            cell_value = str(row.get(col, "")).lower()
                            if keyword.lower() in cell_value:
                                content = self._create_document_content(row)
                                metadata = {
                                    "row_index": idx,
                                    "product_name": row.get("제품명", ""),
                                    "company_name": row.get("업체명", ""),
                                    "main_ingredient": row.get("주성분", ""),
                                    "match_type": "exact"
                                }
                                doc = Document(page_content=content, metadata=metadata)
                                exact_results.append((doc, 1.0))  # 정확한 매칭 가중치 1.0
                                break
            
            # 4. 결과 통합 및 정렬
            all_results = vector_results + keyword_results + exact_results
            
            # 중복 제거 (row_index 기준)
            seen_indices = set()
            unique_results = []
            
            for doc, weight in all_results:
                row_idx = doc.metadata.get("row_index", -1)
                if row_idx not in seen_indices:
                    seen_indices.add(row_idx)
                    unique_results.append((doc, weight))
            
            # 가중치로 정렬
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 결과 반환
            final_results = [doc for doc, weight in unique_results[:k]]
            
            logger.info(f"가중치 검색 결과: {len(final_results)}개")
            return final_results
            
        except Exception as e:
            logger.error(f"가중치 검색 오류: {e}")
            return []
    
    def _create_document_content(self, row):
        """행 데이터로부터 문서 내용 생성"""
        important_content = []
        other_content = []
        
        for col in self.df.columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                content = f"{col}: {row[col]}"
                
                if col in self.important_columns:
                    important_content.append(content)
                else:
                    other_content.append(content)
        
        if important_content:
            content = "\n".join(important_content)
            if other_content:
                content += "\n\n" + "\n".join(other_content)
        else:
            content = "\n".join(other_content)
        
        return content
    
    def build_vectorstore(self):
        """벡터 데이터베이스 구축"""
        try:
            df = self.load_csv_data()
            
            # 개선된 Document 객체 생성
            documents = self.create_improved_documents_from_csv(df)
            
            if not documents:
                raise ValueError("생성된 문서가 없습니다. CSV 파일을 확인해주세요.")
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,  # 더 큰 청크로 설정
                chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(documents)
            
            logger.info(f"텍스트 분할 완료: {len(doc_splits)} 개 청크")
            
            if not doc_splits:
                raise ValueError("분할된 문서가 없습니다.")
            
            # 벡터 데이터베이스 생성
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="improved-medicine-rag-chroma",
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            
            if not hasattr(self.vectorstore, 'as_retriever'):
                raise Exception("벡터스토어 생성 실패: as_retriever 메서드 없음")
            logger.info("벡터 데이터베이스 구축 완료")
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스 구축 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            self.vectorstore = None
            raise
    
    def load_existing_vectorstore(self):
        """기존 벡터 데이터베이스 로드"""
        try:
            self.vectorstore = Chroma(
                collection_name="improved-medicine-rag-chroma",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            if not hasattr(self.vectorstore, 'as_retriever'):
                raise Exception("벡터스토어 로드 실패: as_retriever 메서드 없음")
            logger.info("기존 벡터 데이터베이스 로드 완료")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"기존 벡터 데이터베이스 로드 실패: {e}")
            self.vectorstore = None
            return None
    
    def load_vectorstore(self):
        """벡터 데이터베이스 로드 (load_existing_vectorstore의 별칭)"""
        return self.load_existing_vectorstore()

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    search_results: List[str]
    keywords: List[str]

def create_improved_medicine_rag_agent(csv_rag: ImprovedCSVMedicineRAG):
    """향상된 의약품 RAG 에이전트 생성"""
    
    def search_node(state: AgentState) -> AgentState:
        try:
            query = state["query"]
            logger.info(f"검색 쿼리: {query}")
            
            # 키워드 추출
            keywords = csv_rag.extract_keywords(query)
            state["keywords"] = keywords
            
            if csv_rag.vectorstore is None:
                raise ValueError("벡터스토어가 초기화되지 않았습니다.")
            
            # 가중치 검색 수행
            docs = csv_rag.weighted_search(query, keywords, k=5)
            context = [doc.page_content for doc in docs]
            
            logger.info(f"검색된 문서 수: {len(docs)}")
            if docs:
                logger.info(f"첫 번째 문서 샘플: {docs[0].page_content[:200]}...")
            
            return {
                "query": query,
                "context": context,
                "search_results": context,
                "answer": "",
                "keywords": keywords
            }
        except Exception as e:
            logger.error(f"검색 노드 오류: {e}")
            import traceback
            logger.error(f"검색 상세 오류: {traceback.format_exc()}")
            return {
                "query": query,
                "context": [],
                "search_results": [],
                "answer": f"[검색 오류] {e}",
                "keywords": []
            }

    def answer_node(state: AgentState) -> AgentState:
        try:
            query = state["query"]
            context = state["context"]
            keywords = state["keywords"]
            
            if csv_rag.llm is None:
                raise ValueError("LLM이 초기화되지 않았습니다.")
            
            # 컨텍스트가 비어있는 경우 처리
            if not context:
                return {
                    "query": query,
                    "context": context,
                    "search_results": state["search_results"],
                    "answer": f"죄송합니다. '{', '.join(keywords)}'와 관련된 의약품 정보를 찾을 수 없습니다. 다른 키워드로 다시 질문해주세요.",
                    "keywords": keywords
                }
            
            prompt_template = PromptTemplate(
                input_variables=["context", "question", "keywords"],
                template="""
                다음 의약품 정보를 기반으로 질문에 답변해주세요.
                의약품 정보는 정확하고 신뢰할 수 있는 정보를 제공해야 합니다.
                
                검색 키워드: {keywords}
                
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
                    question=query,
                    keywords=", ".join(keywords)
                )
            )
            
            return {
                "query": query,
                "context": context,
                "search_results": state["search_results"],
                "answer": response,
                "keywords": keywords
            }
        except Exception as e:
            logger.error(f"답변 노드 오류: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                "query": state.get("query", ""),
                "context": state.get("context", []),
                "search_results": state.get("search_results", []),
                "answer": f"[답변 오류] {e}",
                "keywords": state.get("keywords", [])
            }

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()

def query_improved_medicine_rag(query: str, csv_rag: ImprovedCSVMedicineRAG):
    """향상된 의약품 RAG 시스템에 질문"""
    agent = create_improved_medicine_rag_agent(csv_rag)
    initial_state: AgentState = {
        "query": query,
        "context": [],
        "answer": "",
        "search_results": [],
        "keywords": []
    }
    result = agent.invoke(initial_state)
    return result

if __name__ == "__main__":
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    try:
        # 향상된 CSV RAG 시스템 초기화
        csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
        
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
            "특정 업체의 제품을 찾고 싶어요",
            "아스피린에 대한 정보를 알려주세요"
        ]
        
        for query in test_queries:
            logger.info(f"\n질문: {query}")
            result = query_improved_medicine_rag(query, csv_rag)
            logger.info(f"추출된 키워드: {result['keywords']}")
            logger.info(f"답변: {result['answer']}")
            logger.info(f"참고 문서 수: {len(result['context'])}")
            
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}") 