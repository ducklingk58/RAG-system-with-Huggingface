#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색 기능만 테스트
"""

import pandas as pd
from improved_csv_rag_system import ImprovedCSVMedicineRAG

def main():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    print("=== 검색 기능 테스트 ===")
    
    try:
        # 1. CSV 데이터 직접 검색
        print("1. CSV 데이터 직접 검색...")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        # 아스피린 관련 데이터 검색
        aspirin_data = df[df.apply(lambda row: row.astype(str).str.contains('아스피린', case=False, na=False).any(), axis=1)]
        print(f"✅ CSV에서 아스피린 관련 데이터: {len(aspirin_data)}개")
        
        if len(aspirin_data) > 0:
            print("\n아스피린 제품 샘플:")
            for idx, row in aspirin_data.head(3).iterrows():
                print(f"  - {row['제품명']} ({row['업체명']}) - {row['주성분']}")
        
        # 2. 시스템 초기화
        print("\n2. RAG 시스템 초기화...")
        csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
        
        # 3. 벡터 DB 로드
        print("3. 벡터 DB 로드...")
        csv_rag.load_vectorstore()
        
        if csv_rag.vectorstore is None:
            print("벡터 DB가 없습니다. 새로 구축합니다...")
            csv_rag.build_vectorstore()
        
        # 4. 키워드 추출 테스트
        print("4. 키워드 추출 테스트...")
        query = "아스피린에 대해서 알려주세요"
        keywords = csv_rag.extract_keywords(query)
        print(f"질문: {query}")
        print(f"추출된 키워드: {keywords}")
        
        # 5. 간단한 벡터 검색
        print("5. 벡터 검색 테스트...")
        try:
            retriever = csv_rag.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            docs = retriever.invoke("아스피린")
            print(f"✅ 벡터 검색 결과: {len(docs)}개")
            
            if docs:
                print("\n검색 결과 샘플:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n--- 결과 {i} ---")
                    print(f"내용: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        print(f"메타데이터: {doc.metadata}")
            else:
                print("❌ 검색 결과가 없습니다.")
                
        except Exception as e:
            print(f"❌ 벡터 검색 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
        
        # 6. 가중치 검색 테스트
        print("\n6. 가중치 검색 테스트...")
        try:
            docs = csv_rag.weighted_search(query, keywords, k=5)
            print(f"✅ 가중치 검색 결과: {len(docs)}개")
            
            if docs:
                print("\n가중치 검색 결과 샘플:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n--- 결과 {i} ---")
                    print(f"내용: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        print(f"메타데이터: {doc.metadata}")
            else:
                print("❌ 가중치 검색 결과가 없습니다.")
                
        except Exception as e:
            print(f"❌ 가중치 검색 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
        
    except Exception as e:
        print(f"❌ 전체 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 