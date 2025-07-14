#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 검색 테스트
"""

from improved_csv_rag_system import ImprovedCSVMedicineRAG

def main():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    print("=== 간단한 검색 테스트 ===")
    
    try:
        # 시스템 초기화
        print("1. 시스템 초기화...")
        csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
        
        # 벡터 DB 로드
        print("2. 벡터 DB 로드...")
        csv_rag.load_vectorstore()
        
        if csv_rag.vectorstore is None:
            print("벡터 DB가 없습니다. 새로 구축합니다...")
            csv_rag.build_vectorstore()
        
        # 키워드 추출 테스트
        print("3. 키워드 추출 테스트...")
        query = "아스피린에 대해서 알려주세요"
        keywords = csv_rag.extract_keywords(query)
        print(f"질문: {query}")
        print(f"추출된 키워드: {keywords}")
        
        # 간단한 벡터 검색 테스트
        print("4. 벡터 검색 테스트...")
        try:
            retriever = csv_rag.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke("아스피린")
            print(f"검색 결과: {len(docs)}개")
            if docs:
                print(f"첫 번째 결과: {docs[0].page_content[:200]}...")
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
        
        # 가중치 검색 테스트
        print("5. 가중치 검색 테스트...")
        try:
            docs = csv_rag.weighted_search(query, keywords, k=3)
            print(f"가중치 검색 결과: {len(docs)}개")
            if docs:
                print(f"첫 번째 결과: {docs[0].page_content[:200]}...")
        except Exception as e:
            print(f"가중치 검색 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 