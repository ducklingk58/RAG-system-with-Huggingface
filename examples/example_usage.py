#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 의약품 RAG 시스템 사용 예제
"""

import os
from csv_rag_system import CSVMedicineRAG, query_medicine_rag, change_model

def main():
    # 기본 CSV 파일 경로 설정
    default_csv_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    # CSV 파일 경로 입력 (기본값 제공)
    csv_file_path = input(f"CSV 파일 경로를 입력하세요 (기본값: {default_csv_path}): ").strip()
    
    # 입력이 없으면 기본값 사용
    if not csv_file_path:
        csv_file_path = default_csv_path
    
    if not os.path.exists(csv_file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    
    try:
        print("의약품 RAG 시스템 초기화 중...")
        
        # CSV RAG 시스템 초기화
        csv_rag = CSVMedicineRAG(csv_file_path)
        
        # 기존 벡터 데이터베이스가 있는지 확인
        if not csv_rag.load_existing_vectorstore():
            print("새로운 벡터 데이터베이스 구축 시작...")
            csv_rag.build_vectorstore()
            print("벡터 데이터베이스 구축 완료!")
        else:
            print("기존 벡터 데이터베이스 로드 완료!")
        
        print("\n=== 의약품 정보 검색 시스템 ===")
        print("질문을 입력하세요 (종료하려면 'quit' 입력)")
        
        while True:
            query = input("\n질문: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not query:
                print("질문을 입력해주세요.")
                continue
            
            try:
                print("검색 중...")
                result = query_medicine_rag(query, csv_rag)
                
                print(f"\n답변: {result['answer']}")
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

def test_queries():
    """테스트 질문들"""
    test_questions = [
        "아스피린에 대한 정보를 알려주세요",
        "두통약 중에서 어떤 것들이 있나요?",
        "항생제 종류에는 어떤 것들이 있나요?",
        "특정 업체의 제품을 찾고 싶어요",
        "알레르기 치료제는 어떤 것들이 있나요?",
        "소화제 종류를 알려주세요",
        "혈압약에는 어떤 것들이 있나요?",
        "당뇨병 치료제 정보를 알려주세요"
    ]
    
    print("=== 테스트 질문 예시 ===")
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. {question}")

if __name__ == "__main__":
    print("=== CSV 의약품 RAG 시스템 ===")
    print("이 시스템은 CSV 파일의 의약품 데이터를 기반으로 질문에 답변합니다.")
    
    # 테스트 질문 예시 표시
    test_queries()
    
    # 메인 실행
    main() 