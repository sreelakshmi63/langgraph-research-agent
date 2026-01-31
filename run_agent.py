import sys
from langchain_core.messages import HumanMessage
from graph import app

from dotenv import load_dotenv
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print('WRONG SYNTAX! Please check the usage: python run_agent.py "your question"')
        return

    question = sys.argv[1]

    print("\n🧠 Question:")
    print(question)
    print("\n🤖 Agent thinking...\n")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
             "steps": 0
        },
        config={"recursion_limit": 100}
    )

    print("\n✅ Final Answer:")
    # print(result["messages"][-1].content)
    print(result)
    print("======================")
    print("======================")
    print("======================")
    print(result["messages"][-1])

if __name__ == "__main__":
    main()
