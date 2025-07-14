"""
LangGraph 노드 함수 예외처리 예시

이 예시는 LangGraph 워크플로우 노드(search_node, answer_node 등)에서
try/except로 오류를 잡아 answer에 에러 메시지를 반환하는 구조를 보여줍니다.
"""

def search_node(state):
    try:
        # 실제 검색 로직 (예시)
        raise ValueError("테스트용 오류!")
        return {
            "query": state["query"],
            "context": ["문서1", "문서2"],
            "search_results": ["문서1", "문서2"],
            "answer": ""
        }
    except Exception as e:
        return {
            "query": state.get("query", ""),
            "context": [],
            "search_results": [],
            "answer": f"[search_node 오류] {e}"
        }

def answer_node(state):
    try:
        # 실제 답변 생성 로직 (예시)
        raise RuntimeError("답변 생성 중 오류!")
        return {
            "query": state["query"],
            "context": state["context"],
            "search_results": state["search_results"],
            "answer": "정상 답변"
        }
    except Exception as e:
        return {
            "query": state.get("query", ""),
            "context": state.get("context", []),
            "search_results": state.get("search_results", []),
            "answer": f"[answer_node 오류] {e}"
        }

if __name__ == "__main__":
    # 테스트
    state = {"query": "예시 질문"}
    print(search_node(state))
    print(answer_node(state)) 