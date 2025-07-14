#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 DB 상태 확인 디버그 스크립트
"""

import os
from improved_csv_rag_system import ImprovedCSVMedicineRAG

def main():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    print("=== 벡터 DB 상태 확인 ===")
    
    try:
        # 시스템 초기화
        print("1. 시스템 초기화...")
        csv_rag = ImprovedCSVMedicineRAG(csv_file_path)
        print("✓ 시스템 초기화 완료")
        
        # 벡터 DB 로드 시도
        print("\n2. 기존 벡터 DB 로드 시도...")
        vectorstore = csv_rag.load_vectorstore()
        
        if vectorstore is None:
            print("✗ 기존 벡터 DB가 없습니다.")
            print("\n3. 새로운 벡터 DB 구축...")
            csv_rag.build_vectorstore()
            print("✓ 새로운 벡터 DB 구축 완료")
        else:
            print("✓ 기존 벡터 DB 로드 완료")
        
        # 벡터 DB 상태 확인
        print("\n4. 벡터 DB 상태 확인...")
        if csv_rag.vectorstore is not None:
            print(f"✓ 벡터스토어 객체: {type(csv_rag.vectorstore)}")
            print(f"✓ as_retriever 메서드 존재: {hasattr(csv_rag.vectorstore, 'as_retriever')}")
            
            # 간단한 검색 테스트
            print("\n5. 검색 테스트...")
            try:
                retriever = csv_rag.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                test_docs = retriever.invoke("아스피린")
                print(f"✓ 검색 테스트 성공: {len(test_docs)}개 결과")
            except Exception as e:
                print(f"✗ 검색 테스트 실패: {e}")
        else:
            print("✗ 벡터스토어가 None입니다.")
        
        print("\n=== 디버그 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 