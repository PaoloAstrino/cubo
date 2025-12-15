
import pytest
import requests
import time
import os
from pathlib import Path

BACKEND_URL = "http://localhost:8000"

def test_chat_persistence_flow():
    """
    Verify backend chat persistence loop:
    1. Send a query -> should get conversation_id
    2. Retrieve history -> should see user query and assistant response
    3. List conversations -> should see the new conversation
    4. Send follow-up -> should append to same conversation
    """
    
    # 0. Wait for backend health
    print("Checking backend health...")
    for _ in range(30):
        try:
            health = requests.get(f"{BACKEND_URL}/api/health", timeout=2)
            if health.status_code == 200:
                print("Backend is healthy.")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.fail("Backend not responding")

    # 0.5 Ensure Index is not empty (required to query)
    # Create a dummy file
    dummy_file_path = Path("dummy_test_doc.txt")
    dummy_file_path.write_text("This is a dummy document for testing persistence.")
    
    try:
        # Upload
        print("Uploading dummy document...")
        with open(dummy_file_path, "rb") as f:
            up_resp = requests.post(f"{BACKEND_URL}/api/upload", files={"file": f})
        assert up_resp.status_code == 200, f"Upload failed: {up_resp.text}"

        # Ingest and Index
        print("Building index...")
        build_resp = requests.post(f"{BACKEND_URL}/api/build-index", json={"force_rebuild": True})
        assert build_resp.status_code == 200, f"Build index failed: {build_resp.text}"
        
        # Wait for index to be ready
        print("Waiting for index readiness...")
        time.sleep(3)
        
    finally:
        # Cleanup file locally (API keeps it)
        if dummy_file_path.exists():
            dummy_file_path.unlink()

    # 1. Send initial query
    query_payload = {
        "query": "Hello, this is a persistence test.",
        "top_k": 3,
        "use_reranker": False # Faster
    }
    print(f"\nSending query to {BACKEND_URL}/api/query...")
    response = requests.post(f"{BACKEND_URL}/api/query", json=query_payload)
    assert response.status_code == 200, f"Query failed: {response.text}"
    data = response.json()
    
    assert "answer" in data
    assert "conversation_id" in data, "Response missing conversation_id"
    conversation_id = data["conversation_id"]
    print(f"Received conversation_id: {conversation_id}")
    
    # 2. Retrieve history
    history_url = f"{BACKEND_URL}/api/chat/history/{conversation_id}"
    print(f"Fetching history from {history_url}...")
    # Give a tiny bit of time for SQLite write/commit if async (though it's synchronous in code)
    # But just in case of any weird race in test environment
    time.sleep(0.5) 
    
    hist_response = requests.get(history_url)
    assert hist_response.status_code == 200, f"Get history failed: {hist_response.text}"
    history_data = hist_response.json()
    
    assert history_data["conversation_id"] == conversation_id
    messages = history_data["messages"]
    assert len(messages) >= 2, "Should have at least user and assistant messages"
    
    # Check content
    user_msgs = [m for m in messages if m["role"] == "user"]
    assist_msgs = [m for m in messages if m["role"] == "assistant"]
    
    assert len(user_msgs) >= 1
    assert user_msgs[0]["content"] == query_payload["query"]
    assert len(assist_msgs) >= 1
    # Assistant msg should match answer
    assert assist_msgs[0]["content"] == data["answer"]
    
    # 3. List conversations
    list_url = f"{BACKEND_URL}/api/chat/conversations"
    print(f"Listing conversations from {list_url}...")
    list_response = requests.get(list_url)
    assert list_response.status_code == 200
    conversations = list_response.json()
    
    # Ensure our conversation is in the list
    found = any(c["id"] == conversation_id for c in conversations)
    assert found, f"Conversation {conversation_id} not found in listing"
    
    # 4. Follow-up query
    follow_up_payload = {
        "query": "What was my previous message?",
        "conversation_id": conversation_id
    }
    print("Sending follow-up query...")
    follow_resp = requests.post(f"{BACKEND_URL}/api/query", json=follow_up_payload)
    assert follow_resp.status_code == 200
    follow_data = follow_resp.json()
    
    assert follow_data["conversation_id"] == conversation_id
    
    # Verify history grew
    hist_response_2 = requests.get(history_url)
    messages_2 = hist_response_2.json()["messages"]
    assert len(messages_2) >= len(messages) + 2 # +1 user, +1 assistant
    
    print("\nBackend Persistence Verified Successfully!")
