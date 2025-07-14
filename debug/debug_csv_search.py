#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 시스템 디버깅
"""

import pandas as pd
import re
from csv_direct_search import CSVDirectMedicineSearch

def debug_csv_search():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    print("=== CSV 직접 검색 시스템 디버깅 ===")
    
    try:
        # 1. 시스템 초기화 테스트
        print("1. 시스템 초기화 테스트...")
        csv_search = CSVDirectMedicineSearch(csv_file_path)
        print(f"✅ 데이터 로드 완료: {len(csv_search.df)} 행")
        
        # 2. 키워드 추출 테스트
        print("\n2. 키워드 추출 테스트...")
        test_queries = [
            "아스피린에 대해서 알려주세요",
            "타이레놀의 주성분이 뭐야?",
            "보건용마스크 중에서 어떤 것들이 있나요?",
            "aspirin에 대한 정보를 알려주세요",
            "엑스원에이정에 대한 정보를 알려주세요"
        ]
        
        for query in test_queries:
            keywords = csv_search.extract_keywords(query)
            print(f"질문: {query}")
            print(f"키워드: {keywords}")
        
        # 3. 검색 성능 테스트
        print("\n3. 검색 성능 테스트...")
        test_keywords = ["아스피린", "타이레놀", "마스크", "aspirin", "엑스원에이정"]
        
        for keyword in test_keywords:
            # 직접 pandas 검색
            mask = csv_search.df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
            count = mask.sum()
            print(f"키워드 '{keyword}': {count}개 결과")
            
            if count > 0:
                sample_data = csv_search.df[mask].head(2)
                for idx, row in sample_data.iterrows():
                    print(f"  - {row.get('제품명', 'N/A')} ({row.get('업체명', 'N/A')})")
        
        # 4. 데이터 품질 확인
        print("\n4. 데이터 품질 확인...")
        
        # 빈 값 확인
        empty_counts = {}
        for col in csv_search.df.columns:
            empty_count = csv_search.df[col].isna().sum()
            if empty_count > 0:
                empty_counts[col] = empty_count
        
        print("빈 값이 있는 컬럼:")
        for col, count in empty_counts.items():
            print(f"  • {col}: {count}개 ({count/len(csv_search.df)*100:.1f}%)")
        
        # 5. 중복 데이터 확인
        print("\n5. 중복 데이터 확인...")
        duplicate_licenses = csv_search.df['허가번호'].duplicated().sum()
        print(f"중복 허가번호: {duplicate_licenses}개")
        
        # 6. 메모리 사용량 확인
        print("\n6. 메모리 사용량 확인...")
        memory_usage = csv_search.df.memory_usage(deep=True).sum() / (1024*1024)
        print(f"메모리 사용량: {memory_usage:.2f} MB")
        
        # 7. 실제 검색 테스트
        print("\n7. 실제 검색 테스트...")
        test_query = "아스피린에 대해서 알려주세요"
        result = csv_search.search_medicine_data(test_query)
        
        print(f"검색 결과:")
        print(f"  • 키워드: {result['keywords']}")
        print(f"  • 총 결과 수: {result['total_count']}")
        print(f"  • 보고서 길이: {len(result['report'])} 문자")
        
        # 8. 성능 측정
        print("\n8. 성능 측정...")
        import time
        
        start_time = time.time()
        result = csv_search.search_medicine_data("아스피린에 대해서 알려주세요")
        end_time = time.time()
        
        print(f"검색 시간: {(end_time - start_time)*1000:.2f}ms")
        
        # 9. 오류 처리 테스트
        print("\n9. 오류 처리 테스트...")
        
        # 빈 질문
        empty_result = csv_search.search_medicine_data("")
        print(f"빈 질문 처리: {empty_result['total_count']}개 결과")
        
        # 존재하지 않는 키워드
        not_found_result = csv_search.search_medicine_data("존재하지않는약물명")
        print(f"존재하지 않는 키워드: {not_found_result['total_count']}개 결과")
        
        print("\n=== 디버깅 완료 ===")
        
    except Exception as e:
        print(f"❌ 디버깅 중 오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_csv_search() 