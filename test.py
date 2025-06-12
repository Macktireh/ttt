import json

import httpx

BASE_URL = "http://localhost:8000"


def test_models():
    """Test manuel des modèles."""
    with httpx.Client() as client:
        response = client.get(f"{BASE_URL}/v1/models")
        print(f"Models: {response.status_code}")
        data = response.json()
        print(data)


def test_chat_simple():
    """Test manuel de chat simple."""
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Bonjour, comment vous appelez-vous ?"},
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis un assistant IA. Comment puis-je vous aider aujourd'hui ?",  # noqa: E501
            },
            {"role": "user", "content": "Parfait ! Pouvez-vous m'aider avec du code Python ?"},
        ],
        "stream": False,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        print(f"Chat: {response.status_code}")
        if response.status_code == 201:
            data = response.json()
            print(f"Réponse: {data['choices'][0]['message']['content']}")


def test_stream_simple():
    """Test manuel de streaming simple."""
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Bonjour, comment vous appelez-vous ?"},
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis un assistant IA. Comment puis-je vous aider aujourd'hui ?",  # noqa: E501
            },
            {"role": "user", "content": "Parfait ! Pouvez-vous m'aider avec du code Python ?"},
        ],
        "stream": True,
    }

    with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60) as r:
        print(f"Stream Status: {r.status_code}")
        for text in r.iter_text():
            if text.strip():
                lines = text.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            print("\n[DONE]")
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                        except Exception:
                            pass


if __name__ == "__main__":
    test_stream_simple()
