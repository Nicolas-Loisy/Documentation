**Openai**
```
import json
import requests

api_key = "sk-********"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Bonjour !"}]
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    data=json.dumps(data)
)

print(response.json()['choices'][0]['message']['content'])

```

**LeChat**
```
import requests

headers = {
    "Authorization": "Bearer KI***********",
    "Content-Type": "application/json"
}

data = {
    "model": "mistral-medium",
    "messages": [
        {"role": "user", "content": "Bonjour, peux-tu m'expliquer la relativité restreinte ?"}
    ]
}

response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])
```
**Avec historique : **
```
import json, requests
from pathlib import Path

API_KEY = "**********"  # Remplace par ta clé
HISTORY_FILE = Path("chat.json")
MODEL = "mistral-medium"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat(msg):
    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    history.append({"role": "user", "content": msg})
    res = requests.post("https://api.mistral.ai/v1/chat/completions",
        headers=headers, json={"model": MODEL, "messages": history})
    reply = res.json()["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return reply

print(chat("Explique les llm"))
print(chat("Redonne ma derniere question"))

```