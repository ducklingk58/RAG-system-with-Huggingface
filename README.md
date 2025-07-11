# LangGraph 기반 RAG Agent 시스템

의약품 안전성 관련 질문에 답변하는 LangGraph 기반 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 특징

- **무료 모델 사용**: Hugging Face의 무료 모델들을 사용하여 API 키 없이도 작동
- **LangGraph 기반**: 복잡한 워크플로우를 그래프로 관리
- **의약품 전문**: 의약품 안전처 및 관련 사이트의 정보를 참고 자료로 활용
- **한국어 지원**: 한국어 질문과 답변 지원

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. (선택사항) Hugging Face API 키 설정:
   - `API_KEY.txt` 파일에 Hugging Face API 키를 저장
   - API 키가 없어도 기본 모델들은 작동합니다

## 사용 가능한 모델

- `flan-t5-small`: 가장 안정적이고 빠른 모델 (기본값)
- `flan-t5-base`: 더 정확한 답변을 제공하는 모델
- `microsoft-dialo`: 대화형 모델
- `gpt2`: OpenAI의 GPT-2 모델
- `distilgpt2`: GPT-2의 경량화 버전

## 사용법

### 1. 기본 테스트
```bash
python rag_agent_example.py
```

### 2. 대화형 모드
```bash
python rag_agent_example.py --interactive
```

### 3. 모델 비교 테스트
```bash
python rag_agent_example.py --test-models
```

### 4. 직접 사용
```python
from rag_system import query_rag_agent, change_model

# 질문하기
result = query_rag_agent("의약품 안전성에 대해 알려주세요")
print(result['answer'])

# 모델 변경
change_model("flan-t5-base")
```

## 시스템 구조

```
입력 쿼리 → 검색 노드 → 관련 문서 검색 → 답변 생성 노드 → 최종 답변
```

1. **검색 노드**: 쿼리를 벡터화하여 관련 문서를 검색
2. **답변 생성 노드**: 검색된 문서를 기반으로 답변 생성

## 참고 자료 URL

시스템에서 사용하는 의약품 관련 웹사이트들:
- https://nedrug.mfds.go.kr/index (의약품정보)
- https://www.health.kr/ (건강정보)
- https://www.kdra.or.kr/website/index.php (한국약물정보)
- https://www.druginfo.co.kr/ (약물정보)
- https://www.mfds.go.kr/ (식약처)

## 파일 구조

```
ddd/
├── rag_system.py          # 메인 RAG 시스템
├── rag_agent_example.py   # 사용 예시
├── requirements.txt        # 필요한 패키지 목록
├── README.md              # 이 파일
└── chroma_db/            # 벡터 데이터베이스 (자동 생성)
```

## 주의사항

1. **인터넷 연결**: 웹 문서를 로드하기 위해 인터넷 연결이 필요합니다
2. **처음 실행 시 시간**: 모델 다운로드로 인해 처음 실행 시 시간이 걸릴 수 있습니다
3. **메모리 사용량**: 큰 모델 사용 시 메모리 사용량이 증가할 수 있습니다

## 문제 해결

### 모델 다운로드 오류
- 인터넷 연결을 확인하세요
- Hugging Face API 키를 설정해보세요

### 메모리 부족 오류
- 더 작은 모델을 사용하세요 (flan-t5-small)
- `device='cpu'` 설정을 확인하세요

### 웹 문서 로드 오류
- URL이 유효한지 확인하세요
- 네트워크 연결을 확인하세요

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 