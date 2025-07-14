#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 파일 데이터 구조 확인
"""

import pandas as pd
import os

def main():
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    print("=== CSV 파일 데이터 확인 ===")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_file_path):
            print(f"❌ 파일이 존재하지 않습니다: {csv_file_path}")
            return
        
        print(f"✅ 파일 경로: {csv_file_path}")
        print(f"✅ 파일 크기: {os.path.getsize(csv_file_path) / (1024*1024):.2f} MB")
        
        # CSV 파일 읽기
        print("\n1. CSV 파일 읽기...")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        print(f"✅ 총 행 수: {len(df)}")
        print(f"✅ 총 열 수: {len(df.columns)}")
        
        # 컬럼명 확인
        print("\n2. 컬럼명 확인:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # 데이터 샘플 확인
        print("\n3. 처음 5행 데이터:")
        print(df.head())
        
        # 아스피린 관련 데이터 검색
        print("\n4. '아스피린' 관련 데이터 검색:")
        aspirin_data = df[df.apply(lambda row: row.astype(str).str.contains('아스피린', case=False, na=False).any(), axis=1)]
        print(f"✅ '아스피린' 관련 데이터: {len(aspirin_data)}개")
        
        if len(aspirin_data) > 0:
            print("\n아스피린 관련 데이터 샘플:")
            for idx, row in aspirin_data.head(3).iterrows():
                print(f"\n--- 행 {idx} ---")
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        print(f"  {col}: {row[col]}")
        
        # 제품명 컬럼 확인
        if '제품명' in df.columns:
            print(f"\n5. 제품명 컬럼 통계:")
            print(f"  - 고유 제품명 수: {df['제품명'].nunique()}")
            print(f"  - 빈 값 수: {df['제품명'].isna().sum()}")
            print(f"  - 샘플 제품명들:")
            sample_products = df['제품명'].dropna().head(10).tolist()
            for i, product in enumerate(sample_products, 1):
                print(f"    {i}. {product}")
        
        # 주성분 컬럼 확인
        if '주성분' in df.columns:
            print(f"\n6. 주성분 컬럼 통계:")
            print(f"  - 고유 주성분 수: {df['주성분'].nunique()}")
            print(f"  - 빈 값 수: {df['주성분'].isna().sum()}")
            print(f"  - 샘플 주성분들:")
            sample_ingredients = df['주성분'].dropna().head(10).tolist()
            for i, ingredient in enumerate(sample_ingredients, 1):
                print(f"    {i}. {ingredient}")
        
        # 데이터 타입 확인
        print("\n7. 데이터 타입 확인:")
        print(df.dtypes)
        
        # 메모리 사용량
        print(f"\n8. 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 