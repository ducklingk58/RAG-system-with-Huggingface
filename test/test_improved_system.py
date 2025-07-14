#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 CSV 의약품 RAG 시스템 테스트
"""

import os
from improved_csv_rag_system import ImprovedCSVMedicineRAG, query_improved_medicine_rag

def main():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    try:
        print("=== 향상된 의약품 RAG 시스템 ===")
        print("시스템 초기화 중...")
        
        # 향상된 CSV RAG 시스템 초기화
        csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
        
        # 기존 벡터 데이터베이스 로드 (빠른 시작을 위해)
        print("기존 벡터 데이터베이스 로드 중...")
        csv_rag.load_vectorstore()
        
        if csv_rag.vectorstore is None:
            print("기존 벡터 DB가 없습니다. 새로운 벡터 데이터베이스 구축 시작...")
            csv_rag.build_vectorstore()
        else:
            print("기존 벡터 데이터베이스 로드 완료!")
        
        print("\n=== 의약품 정보 검색 시스템 ===")
        print("질문을 입력하세요 (종료하려면 'quit' 입력)")
        print("예시 질문:")
        print("- 엑스원에이정에 대한 정보를 알려주세요")
        print("- 보건용마스크 중에서 어떤 것들이 있나요?")
        print("- 파비안디정에 대해서 알려주세요")
        print("- 아스피린에 대한 정보를 알려주세요")
        
        while True:
            query = input("\n질문: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not query:
                print("질문을 입력해주세요.")
                continue
            
            try:
                if csv_rag.vectorstore is None or not hasattr(csv_rag.vectorstore, 'as_retriever'):
                    print("[오류] 벡터 데이터베이스가 올바르게 초기화되지 않았습니다. 시스템을 재시작하거나 build_vectorstore를 다시 실행하세요.")
                    continue
                print("검색 중...")
                result = query_improved_medicine_rag(query, csv_rag)
                
                print(f"\n추출된 키워드: {result['keywords']}")
                print(f"답변: {result['answer']}")
                print(f"참고 문서 수: {len(result['context'])}")
                
                # 참고 문서 표시 (선택사항)
                show_context = input("\n참고 문서를 보시겠습니까? (y/n): ").strip().lower()
                if show_context == 'y':
                    for i, context in enumerate(result['context'], 1):
                        print(f"\n--- 참고 문서 {i} ---")
                        print(context[:500] + "..." if len(context) > 500 else context)
                
            except Exception as e:
                print(f"오류 발생: {e}")
    
    except Exception as e:
        print(f"시스템 초기화 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 