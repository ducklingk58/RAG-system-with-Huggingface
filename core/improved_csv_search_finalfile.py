#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 CSV 직접 검색 시스템
- 성능 최적화
- 정확한 키워드 추출
- 빠른 검색 속도
"""

import pandas as pd
import re
from typing import List, Dict, Any
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCSVDirectSearch:
    def __init__(self, csv_file_path: str):
        """개선된 CSV 직접 검색 시스템 초기화"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.search_cache = {}  # 검색 결과 캐시
        self.load_csv_data()
    
    def load_csv_data(self):
        """CSV 파일 로드 및 최적화"""
        try:
            logger.info(f"CSV 파일 로딩 중: {self.csv_file_path}")
            self.df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            
            # 데이터 전처리
            self.df = self.df.fillna("")
            
            # 중복 제거 (허가번호 기준)
            self.df = self.df.drop_duplicates(subset=['허가번호'], keep='first')
            
            # 검색용 컬럼 생성 (빠른 검색을 위해)
            self.df['search_text'] = (
                self.df['제품명'].astype(str) + " " +
                self.df['주성분'].astype(str) + " " +
                self.df['업체명'].astype(str) + " " +
                self.df['제품영문명'].astype(str) + " " +
                self.df['주성분영문'].astype(str)
            ).str.lower()
            
            logger.info(f"데이터 로딩 완료: {len(self.df)} 행 (중복 제거 후)")
            
        except Exception as e:
            logger.error(f"CSV 파일 로딩 실패: {e}")
            raise
    
    def extract_keywords(self, query: str) -> List[str]:
        """개선된 키워드 추출"""
        keywords = []
        
        # 1. 의약품 관련 패턴
        medicine_patterns = [
            r'[가-힣]+정',  # ~정
            r'[가-힣]+액',  # ~액
            r'[가-힣]+캡슐',  # ~캡슐
            r'[가-힣]+시럽',  # ~시럽
            r'[가-힣]+주',  # ~주
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
        
        # 패턴 매칭
        for pattern in medicine_patterns:
            matches = re.findall(pattern, query)
            keywords.extend(matches)
        
        # 2. 일반 의약품명 (3글자 이상)
        medicine_names = re.findall(r'[가-힣]{3,}', query)
        keywords.extend(medicine_names)
        
        # 3. 영어 의약품명 (3글자 이상)
        english_names = re.findall(r'[a-zA-Z]{3,}', query)
        keywords.extend(english_names)
        
        # 4. 불필요한 단어 제거
        stop_words = {
            '에', '의', '를', '을', '가', '이', '는', '은', '도', '만', '대한', 
            '정보를', '알려주세요', '어떤', '것들이', '있나요', '대해서', '알려주',
            '뭔가요', '무엇인가요', '알려주세요', '알려줘', '정보', '대해서',
            '뭐야', '무엇', '어떤', '것들', '중에서', '있나요', '알려줘',
            '주성분', '성분', '제품', '약품', '의약품', '제형', '업체', '회사'
        }
        
        # 조사 제거 및 필터링
        cleaned_keywords = []
        for keyword in keywords:
            # 조사 제거
            for stop_word in stop_words:
                if keyword.endswith(stop_word):
                    keyword = keyword[:-len(stop_word)]
                    break
            
            # 3글자 이상이고 stop_words에 없는 경우만 추가
            if len(keyword) >= 3 and keyword not in stop_words:
                cleaned_keywords.append(keyword)
        
        # 중복 제거
        cleaned_keywords = list(set(cleaned_keywords))
        
        logger.info(f"추출된 키워드: {cleaned_keywords}")
        return cleaned_keywords
    
    def fast_search(self, query: str) -> Dict[str, Any]:
        """빠른 검색 수행"""
        start_time = time.time()
        
        try:
            # 키워드 추출
            keywords = self.extract_keywords(query)
            
            if not keywords:
                return {
                    "keywords": [],
                    "total_count": 0,
                    "report": "키워드를 추출할 수 없습니다.",
                    "data": [],
                    "search_time": 0
                }
            
            # 캐시 확인
            cache_key = " ".join(sorted(keywords))
            if cache_key in self.search_cache:
                result = self.search_cache[cache_key].copy()
                result["search_time"] = time.time() - start_time
                return result
            
            # 빠른 검색 (search_text 컬럼 사용)
            search_results = []
            for keyword in keywords:
                mask = self.df['search_text'].str.contains(keyword, case=False, na=False)
                keyword_results = self.df[mask].to_dict('records')
                search_results.extend(keyword_results)
            
            # 중복 제거 (허가번호 기준)
            unique_results = []
            seen_licenses = set()
            for result in search_results:
                license_num = result.get('허가번호', '')
                if license_num not in seen_licenses:
                    seen_licenses.add(license_num)
                    unique_results.append(result)
            
            total_count = len(unique_results)
            
            if total_count == 0:
                result = {
                    "keywords": keywords,
                    "total_count": 0,
                    "report": f"'{', '.join(keywords)}'와 관련된 의약품 정보를 찾을 수 없습니다.",
                    "data": []
                }
            else:
                # 보고서 생성
                report = self.generate_improved_report(unique_results, keywords)
                result = {
                    "keywords": keywords,
                    "total_count": total_count,
                    "report": report,
                    "data": unique_results
                }
            
            # 캐시에 저장
            self.search_cache[cache_key] = result.copy()
            result["search_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return {
                "keywords": [],
                "total_count": 0,
                "report": f"검색 중 오류가 발생했습니다: {e}",
                "data": [],
                "search_time": time.time() - start_time
            }
    
    def generate_improved_report(self, data: List[Dict], keywords: List[str]) -> str:
        """개선된 보고서 생성"""
        if not data:
            return "검색 결과가 없습니다."
        
        report_lines = []
        report_lines.append(f"=== 의약품 검색 보고서 ===")
        report_lines.append(f"검색 키워드: {', '.join(keywords)}")
        report_lines.append(f"총 검색 결과: {len(data)}개")
        report_lines.append("")
        
        # 1. 제품 분류별 통계
        classifications = {}
        for item in data:
            classification = item.get('품목분류', '미분류')
            if classification and classification != '미분류':
                if classification not in classifications:
                    classifications[classification] = 0
                classifications[classification] += 1
        
        if classifications:
            report_lines.append("📊 제품 분류별 통계:")
            for classification, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  • {classification}: {count}개")
            report_lines.append("")
        
        # 2. 업체별 통계
        companies = {}
        for item in data:
            company = item.get('업체명', '미상')
            if company and company != '미상':
                if company not in companies:
                    companies[company] = 0
                companies[company] += 1
        
        if companies:
            report_lines.append("🏢 업체별 통계:")
            for company, count in sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]:
                report_lines.append(f"  • {company}: {count}개")
            report_lines.append("")
        
        # 3. 제형별 통계
        forms = {}
        for item in data:
            form = item.get('제형', '미상')
            if form and form != '미상':
                if form not in forms:
                    forms[form] = 0
                forms[form] += 1
        
        if forms:
            report_lines.append("💊 제형별 통계:")
            for form, count in sorted(forms.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  • {form}: {count}개")
            report_lines.append("")
        
        # 4. 상세 제품 정보 (최대 15개)
        report_lines.append("📋 상세 제품 정보:")
        display_count = min(15, len(data))
        
        for i, item in enumerate(data[:display_count], 1):
            report_lines.append(f"\n{i}. {item.get('제품명', '제품명 없음')}")
            report_lines.append(f"   • 업체: {item.get('업체명', '미상')}")
            
            main_ingredient = item.get('주성분', '')
            if main_ingredient:
                report_lines.append(f"   • 주성분: {main_ingredient}")
            
            form = item.get('제형', '')
            if form:
                report_lines.append(f"   • 제형: {form}")
            
            report_lines.append(f"   • 허가번호: {item.get('허가번호', '미상')}")
            report_lines.append(f"   • 전문/일반: {item.get('전문의약품', '미상')}")
            
            # 추가 정보
            atc_code = item.get('ATC코드', '')
            if atc_code:
                report_lines.append(f"   • ATC코드: {atc_code}")
            
            english_name = item.get('제품영문명', '')
            if english_name:
                report_lines.append(f"   • 영문명: {english_name}")
        
        if len(data) > display_count:
            report_lines.append(f"\n... 외 {len(data) - display_count}개 제품")
        
        return "\n".join(report_lines)

def query_improved_csv_search(query: str, csv_search: ImprovedCSVDirectSearch):
    """개선된 CSV 검색 시스템에 질문"""
    result = csv_search.fast_search(query)
    return result

if __name__ == "__main__":
    # CSV 파일 경로 설정
    csv_file_path = "C:/Users/tel33/OneDrive/바탕 화면/2025년 하계방학/7월 인턴 넥스제너/codes/간단한 DB./Medical_product_data.csv"
    
    try:
        # 개선된 CSV 검색 시스템 초기화
        csv_search = ImprovedCSVDirectSearch(csv_file_path)
        
        print("=== 개선된 CSV 직접 검색 의약품 시스템 ===")
        print("질문을 입력하세요 (종료하려면 'quit' 입력)")
        print("예시 질문:")
        print("- 아스피린에 대해서 알려주세요")
        print("- 타이레놀의 주성분이 뭐야?")
        print("- 보건용마스크 중에서 어떤 것들이 있나요?")
        
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
                result = query_improved_csv_search(query, csv_search)
                
                print(f"\n추출된 키워드: {result['keywords']}")
                print(f"참고 데이터 수: {result['total_count']}")
                print(f"검색 시간: {result.get('search_time', 0)*1000:.2f}ms")
                print(f"\n{result['report']}")
                
            except Exception as e:
                print(f"오류 발생: {e}")
    
    except Exception as e:
        print(f"시스템 초기화 중 오류 발생: {e}") 