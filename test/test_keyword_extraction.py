#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
키워드 추출 테스트 스크립트
"""

from improved_csv_rag_system import ImprovedCSVMedicineRAG

def test_keyword_extraction():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    # 시스템 초기화
    csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
    
    # 테스트 질문들
    test_queries = [
        "아스피린이 뭔가요?",
        "아스피린에 대해서 알려주세요",
        "엑스원에이정에 대한 정보를 알려주세요",
        "보건용마스크 중에서 어떤 것들이 있나요?",
        "파비안디정에 대해서 알려주세요",
        "타이레놀의 주성분이 뭐야?",
        "두통약 중에서 어떤 것들이 있나요?",
        "aspirin에 대한 정보를 알려주세요"
    ]
    
    print("=== 키워드 추출 테스트 ===")
    
    for query in test_queries:
        print(f"\n질문: {query}")
        
        # LLM 키워드 추출 시도
        try:
            llm_keywords = csv_rag.extract_keywords(query)
            print(f"LLM 키워드: {llm_keywords}")
        except Exception as e:
            print(f"LLM 키워드 추출 실패: {e}")
        
        # Fallback 키워드 추출
        fallback_keywords = csv_rag._extract_keywords_fallback(query)
        print(f"Fallback 키워드: {fallback_keywords}")

if __name__ == "__main__":
    test_keyword_extraction() 