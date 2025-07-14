#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 파일 디버깅 스크립트
"""

import pandas as pd
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_csv_file():
    """CSV 파일을 직접 읽어서 내용 확인"""
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_file_path):
            logger.error(f"파일이 존재하지 않습니다: {csv_file_path}")
            return
        
        logger.info(f"CSV 파일 크기: {os.path.getsize(csv_file_path)} bytes")
        
        # CSV 파일 읽기
        logger.info("CSV 파일 읽는 중...")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        logger.info(f"데이터프레임 정보:")
        logger.info(f"- 행 수: {len(df)}")
        logger.info(f"- 열 수: {len(df.columns)}")
        logger.info(f"- 열 이름: {list(df.columns)}")
        
        # 처음 5행 확인
        logger.info("\n처음 5행:")
        print(df.head())
        
        # 제품명 컬럼 확인
        if '제품명' in df.columns:
            logger.info(f"\n제품명 샘플 (처음 10개):")
            print(df['제품명'].head(10).values.tolist())
        
        # 특정 제품 검색
        if '제품명' in df.columns:
            logger.info("\n'엑스원에이정' 검색 결과:")
            result = df[df['제품명'].str.contains('엑스원에이정', case=False, na=False)]
            logger.info(f"검색된 행 수: {len(result)}")
            if len(result) > 0:
                print(result[['제품명', '업체명', '주성분']].head())
        
        # 빈 값 확인
        logger.info(f"\n빈 값 개수:")
        for col in df.columns:
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                logger.info(f"- {col}: {empty_count}개")
        
        # 데이터 타입 확인
        logger.info(f"\n데이터 타입:")
        print(df.dtypes)
        
    except Exception as e:
        logger.error(f"CSV 파일 읽기 오류: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_csv_file() 