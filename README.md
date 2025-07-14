# Improved CSV-Based Medicine Search System

본 프로젝트는 의약품 데이터에 기반한 **빠르고 정확한 직접 검색 시스템**입니다. 
CSV 파일을 기반으로 키워드를 정제·추출하고, 고속 검색을 통해 결과를 도출하며 통계와 요약 정보를 함께 제공합니다.

---

## 🧠 주요 기능

- **정제된 키워드 추출**: 한국어 의약품 표현에 특화된 정규표현식을 사용해 의미 있는 키워드만 필터링합니다.
- **빠른 검색**: 전처리된 검색 컬럼(`search_text`)을 기반으로 고속 검색 수행
- **자동 통계 분석**: 품목 분류, 업체별, 제형별 요약 제공
- **보고서 생성**: 최대 15개의 의약품 상세정보와 함께 요약 리포트 생성
- **검색 캐시 적용**: 동일한 검색 질의에 대해 빠른 응답 속도 제공

---

## 📁 디렉토리 구성 (예시)

```bash
📂 improved-medicine-search
│
├── improved_csv_search_finalfile.py   # 메인 검색 시스템
├── requirements.txt                   # 의존 라이브러리 목록
├── Medical_product_data.csv           # CSV 기반 의약품 데이터 (개별 준비)
└── README.md                          # 프로젝트 설명 파일 (본 문서)
```

---

## 🏁 실행 방법

1. 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

2. 메인 스크립트를 실행합니다.

```bash
python improved_csv_search_finalfile.py
```

3. 콘솔에 질문을 입력해 검색을 수행합니다.

예시:

```
질문: 아스피린에 대해 알려줘
질문: 타이레놀의 주성분은?
질문: 보건용마스크 제품 알려줘
```

---

## 💡 주요 클래스 및 함수 설명

### `ImprovedCSVDirectSearch`
- `__init__`: CSV 경로를 받아 검색 시스템 초기화
- `extract_keywords(query)`: 정규표현식 기반 키워드 추출
- `fast_search(query)`: 키워드 기반 고속 검색 + 캐시
- `generate_improved_report(results, keywords)`: 사용자 친화적 통계 요약 리포트 생성

### `query_improved_csv_search(query, search_instance)`
- 사용자 질문을 입력 받아 시스템에 질의하고 결과 반환

---

## 📝 CSV 데이터 조건

데이터는 반드시 다음 컬럼들을 포함해야 합니다:
- `제품명`, `업체명`, `주성분`, `제품영문명`, `주성분영문`, `제형`, `품목분류`, `허가번호` 등

중복 데이터는 `허가번호` 기준으로 자동 제거됩니다.

---

## 🔧 사용된 기술

- Python 3.x
- pandas
- re (정규표현식)
- logging
- scikit-learn (TF-IDF 추출용, 향후 확장 고려)

---

## 📜 라이선스

본 프로젝트는 개인 학습 및 비상업적 용도에 한해 자유롭게 사용할 수 있습니다.
