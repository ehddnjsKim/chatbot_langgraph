import os
import sys
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# 1. 환경 변수 불러오기
load_dotenv()  # .env 파일에서 OPENAI_API_KEY 읽기
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not found! Please set it in your .env file.")
    sys.exit(1)

# 2. State 정의 
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. LLM 초기화 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. chatbot 노드 
def chatbot(state: State):
    # LLM 호출하여 응답 생성
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5. Graph 구성 
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 6. 실행 함수 
def run_chatbot():
    print("Chatbot started! Type 'quit', 'exit', or 'q' to stop.")

    # 대화 이력을 보관하는 상태
    conversation_state = {"messages": []}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Exiting chatbot. Goodbye!")
            break

        # 사용자 입력을 상태에 추가
        conversation_state["messages"].append(("user", user_input))

        # LLM 호출 (그래프 실행)
        events = graph.stream(conversation_state)

        for event in events:
            for value in event.values():
                if "messages" in value:
                    last_message = value["messages"][-1]
                    print("Bot:", last_message.content)

                    # 모델 응답도 상태에 추가
                    conversation_state["messages"].append(
                        ("assistant", last_message.content)
                    )

if __name__ == "__main__":
    run_chatbot()
