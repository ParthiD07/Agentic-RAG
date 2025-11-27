from dotenv import load_dotenv
load_dotenv()

from graph.graph import app,app1,app2

def main():
    result = app2.invoke(input={"question":"How to prevent LLM from adversarial attacks?"})
    print(result["question"])
    print(result["generation"])
    

if __name__ == "__main__":
    main()
