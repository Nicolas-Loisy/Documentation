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