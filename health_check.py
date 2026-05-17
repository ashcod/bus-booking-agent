import httpx

print("=" * 40)
print("BUSBOT HEALTH CHECK")
print("=" * 40)

# test main app
try:
    r = httpx.get("http://localhost:8000/health", timeout=5)
    print("Main app (8000):    OK -", r.json())
except Exception as e:
    print("Main app (8000):    FAIL -", e)

# test tool server
try:
    r = httpx.get("http://localhost:8001/tools", timeout=5)
    tools = r.json()["tools"]
    print(f"Tool server (8001): OK - {len(tools)} tools")
    for t in tools:
        print(f"  - {t['name']}")
except Exception as e:
    print("Tool server (8001): FAIL -", e)

# test chat endpoint
try:
    r = httpx.post("http://localhost:8000/chat", json={
        "message": "buses from Hyderabad to Bangalore",
        "session_id": "health-check-1",
        "user_id": "test"
    }, timeout=30)
    data = r.json()
    print(f"Chat endpoint:      OK - agent={data['responding_agent']}")
    print(f"Response preview:   {data['response'][:80]}")
except Exception as e:
    print("Chat endpoint:      FAIL -", e)

print("=" * 40)