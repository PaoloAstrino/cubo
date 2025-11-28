
import json
import os

def test_load():
    try:
        with open("dummy_questions.json", encoding="utf-8") as f:
            data = json.load(f)
        print("JSON loaded successfully.")
        print("Keys:", data.keys())
        print("Questions keys:", data.get("questions", {}).keys())
        print("Easy questions:", data.get("questions", {}).get("easy"))
        
        # Simulate the logic in RAGTester
        questions = data["questions"]
        easy = questions.get("easy", [])
        print(f"Easy count: {len(easy)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_load()
