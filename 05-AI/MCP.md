# Model Context Protocol (MCP)

## Qu'est-ce que MCP ?

MCP (Model Context Protocol) est un protocole ouvert qui permet aux LLMs de se connecter à des sources de données et outils externes de manière standardisée.

- **Client** : l'application IA (ex: Claude Desktop, IDE)
- **Server** : expose des outils, ressources ou prompts à l'IA
- **Transport** : communication via stdio ou SSE (HTTP)

## Structure d'un serveur MCP

```
mon-serveur-mcp/
├── src/
│   └── index.ts       # Entrée principale
├── package.json
└── tsconfig.json
```

## Exemple minimal (TypeScript)

```ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "mon-serveur", version: "1.0.0" });

server.tool("dire_bonjour", { nom: z.string() }, async ({ nom }) => ({
  content: [{ type: "text", text: `Bonjour, ${nom} !` }],
}));

const transport = new StdioServerTransport();
await server.connect(transport);
```

## Debugger MCP Inspector

L'inspecteur MCP permet de tester et debugger un serveur MCP localement via une interface web.

```bash
npx @modelcontextprotocol/inspector
```

### Avec un serveur spécifique

```bash
# Serveur Node.js
npx @modelcontextprotocol/inspector node dist/index.js

# Serveur Python
npx @modelcontextprotocol/inspector python server.py
```

### Interface

L'inspecteur démarre sur `http://localhost:5173` (par défaut) et expose :

| Onglet | Description |
|--------|-------------|
| **Tools** | Lister et tester les outils déclarés |
| **Resources** | Explorer les ressources exposées |
| **Prompts** | Tester les prompts définis |
| **Logs** | Voir les échanges JSON-RPC bruts |

### Variables d'environnement

```bash
npx @modelcontextprotocol/inspector \
  -e API_KEY=ma_cle \
  -e DEBUG=true \
  node dist/index.js
```

## Intégration Claude Desktop

Dans `claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "mon-serveur": {
      "command": "node",
      "args": ["/chemin/vers/dist/index.js"],
      "env": {
        "API_KEY": "ma_cle"
      }
    }
  }
}
```

## Ressources

- Spec officielle : [modelcontextprotocol.io](https://modelcontextprotocol.io)
- SDK TypeScript : [@modelcontextprotocol/sdk](https://www.npmjs.com/package/@modelcontextprotocol/sdk)
- Répertoire de serveurs : [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
