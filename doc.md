```
!pip install openai markdown html2docx weasyprint pandas openpyxl python-pptx htmldocx
```

```
"""
Hybrid Structured Document Generator
======================================
Uses an LLM to produce structured JSON, then converts to PDF, DOCX, XLSX, or PPTX.

Key strategy: Markdown -> HTML -> PDF/DOCX (zero manual parsing)

Dependencies:
    pip install anthropic markdown weasyprint htmldocx python-docx python-pptx pandas openpyxl
"""

import json
import markdown
import pandas as pd

# --- PDF ---
from weasyprint import HTML as WeasyHTML

# --- DOCX ---
from docx import Document
from htmldocx import HtmlToDocx

# --- PPTX ---
from pptx import Presentation
from pptx.util import Inches, Pt

# 1. SYSTEM PROMPT — Strict JSON Schema

SYSTEM_PROMPT = """You are a document generation engine.
You must NEVER chat, explain, or add any text outside the JSON.
Your output must be valid JSON (RFC 8259) strictly following ONE of the schemas below based on the user's request.

--- CASE 1: TEXT DOCUMENT (PDF or DOCX) ---
{
  "action": "generate_doc",
  "format": "docx" | "pdf",
  "filename": "suggested_filename",
  "content": {
    "title": "Main Document Title",
    "body_markdown": "Full content in Markdown syntax.\\n\\n# Heading 1\\nParagraph text here.\\n\\n## Subheading\\n- Bullet point 1\\n- Bullet point 2\\n\\n**Bold text** and *italic text* are supported."
  }
}

--- CASE 2: SPREADSHEET (Excel) ---
{
  "action": "generate_doc",
  "format": "xlsx",
  "filename": "suggested_filename",
  "content": {
    "sheet_name": "Sheet1",
    "dataset": [
      {"Column A": "Value 1", "Column B": 100},
      {"Column A": "Value 2", "Column B": 200}
    ]
  }
}

--- CASE 3: PRESENTATION (PowerPoint) ---
{
  "action": "generate_doc",
  "format": "pptx",
  "filename": "suggested_filename",
  "content": {
    "slides": [
      {
        "title": "Slide 1 Title",
        "bullets": ["Key point 1", "Key point 2", "Key point 3"]
      },
      {
        "title": "Slide 2 Title",
        "bullets": ["Another key point", "Conclusion"]
      }
    ]
  }
}

--- CASE 4: Simple conversation (no document needed) ---
{
  "action": "chat",
  "response_text": "Your normal response here."
}

RULES:
- Output ONLY the JSON object. No markdown fences, no preamble, no explanation.
- The "filename" field must not include the file extension.
- For "body_markdown": use full Markdown (headings, bold, italic, lists, tables, code blocks).
- For "dataset": every object in the array must have the exact same keys.
- For "slides": provide 3-8 slides with 3-5 bullets each for typical presentations.
- Choose "chat" ONLY when the user is asking a question that does NOT require a document.
"""


# 2. LLM CALLER — Supports Anthropic & OpenAI-compatible

def call_openai(user_input: str, api_key: str, model: str = "gpt-4o", base_url: str = None) -> dict:
    """Call OpenAI (or any compatible API), return parsed JSON."""
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(response.choices[0].message.content)


# 3. MAIN FUNCTION

def generate_smart_doc(user_input: str, llm_caller, output_dir: str = ".") -> str:
    """
    Send request to LLM, parse JSON, generate file.

    Args:
        user_input: The user's request in natural language.
        llm_caller: A function(user_input) -> dict (parsed JSON).
        output_dir: Directory for generated files.

    Returns:
        Confirmation message or chat response.
    """
    import os

    data = llm_caller(user_input)

    action = data.get("action")

    if action == "chat":
        return f"[CHAT] {data['response_text']}"

    if action == "generate_doc":
        fmt = data["format"]
        content = data["content"]
        fname = data.get("filename", "document")
        filepath = os.path.join(output_dir, f"{fname}.{fmt}")

        builders = {
            "pdf": build_pdf,
            "docx": build_docx,
            "xlsx": build_excel,
            "pptx": build_pptx,
        }

        builder = builders.get(fmt)
        if builder is None:
            return f"Error: format '{fmt}' not recognized."

        builder(content, filepath)
        return f"File generated: {filepath}"

    return "Error: unrecognized action in JSON."


# 4. BUILDERS

def build_pdf(content: dict, filepath: str):
    """Markdown -> HTML + CSS -> PDF via WeasyPrint."""

    title = content.get("title", "Document")
    body_md = content.get("body_markdown", "")

    body_html = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 2.5cm;
    }}
    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #222;
    }}
    h1.doc-title {{
        font-size: 22pt;
        color: #1a1a2e;
        border-bottom: 2px solid #1a1a2e;
        padding-bottom: 8px;
        margin-bottom: 24px;
    }}
    h1 {{ font-size: 18pt; color: #1a1a2e; margin-top: 28px; }}
    h2 {{ font-size: 14pt; color: #2c3e50; margin-top: 22px; }}
    h3 {{ font-size: 12pt; color: #34495e; margin-top: 18px; }}
    ul, ol {{ padding-left: 24px; }}
    li {{ margin-bottom: 4px; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f0f0f0;
        font-weight: bold;
    }}
    code {{
        background: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 10pt;
    }}
    pre {{
        background: #f4f4f4;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
    }}
    blockquote {{
        border-left: 3px solid #ccc;
        margin-left: 0;
        padding-left: 16px;
        color: #555;
    }}
</style>
</head>
<body>
    <h1 class="doc-title">{title}</h1>
    {body_html}
</body>
</html>"""

    WeasyHTML(string=full_html).write_pdf(filepath)


def build_docx(content: dict, filepath: str):
    """Markdown -> HTML -> DOCX via htmldocx."""

    title = content.get("title", "Document")
    body_md = content.get("body_markdown", "")

    body_html = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "toc"],
    )

    doc = Document()
    doc.add_heading(title, level=0)

    parser = HtmlToDocx()
    parser.add_html_to_document(body_html, doc)

    doc.save(filepath)


def build_excel(content: dict, filepath: str):
    """Dataset (list of dicts) -> DataFrame -> .xlsx."""

    dataset = content.get("dataset", [])
    sheet_name = content.get("sheet_name", "Sheet1")

    df = pd.DataFrame(dataset)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Auto-fit column widths
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col_cells)
            col_letter = col_cells[0].column_letter
            ws.column_dimensions[col_letter].width = min(max_len + 3, 50)


def build_pptx(content: dict, filepath: str):
    """Slides (title + bullets) -> PowerPoint."""

    slides_data = content.get("slides", [])
    prs = Presentation()

    for slide_data in slides_data:
        slide_layout = prs.slide_layouts[1]  # Title + Content
        slide = prs.slides.add_slide(slide_layout)

        slide.shapes.title.text = slide_data.get("title", "")

        tf = slide.placeholders[1].text_frame
        tf.clear()

        for i, bullet in enumerate(slide_data.get("bullets", [])):
            if i == 0:
                tf.paragraphs[0].text = bullet
            else:
                p = tf.add_paragraph()
                p.text = bullet
                p.level = 0

    prs.save(filepath)

```

```
# 5. CLI TEST

import sys
import os

output_dir = "./generated"
os.makedirs(output_dir, exist_ok=True)

# --- REAL LLM MODE ---
from google.colab import userdata
API_KEY = userdata.get('apikey')
caller = lambda prompt: call_openai(prompt, api_key=API_KEY)

test_prompts = [
    "Create a PDF report about the 3 main benefits of remote work, with statistics.",
    "Create an Excel spreadsheet listing the top 10 programming languages in 2025 with columns: Rank, Language, Popularity %, Primary Use Case.",
    "Create a PowerPoint presentation about the Apollo space program with 4 slides.",
    "What is the capital of France?",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}: {prompt[:70]}...")
    print(f"{'='*60}")
    try:
        result = generate_smart_doc(prompt, caller, output_dir=output_dir)
        print(f"  -> {result}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\nDone!")
```








```
public override void OnPageRequest()
        {
            // 2. Accès via Hm.Context pour Request et Response
            var context = this.Hm.Context;
            var request = context.Request;
            var response = context.Response;

            // Récupération du paramètre (ex: ?file=mon_doc.pdf)
            string fileName = request.Query["file"];
            
            // Sécurisation basique du chemin (A ADAPTER selon votre dossier réel)
            string baseDir = @"C:\sinequa\data\mon_dossier_partage\";
            string filePath = Path.Combine(baseDir, fileName);

            // Vérification de sécurité (empêcher de remonter dans l'arborescence)
            if (!File.Exists(filePath) || !Path.GetFullPath(filePath).StartsWith(baseDir))
            {
                response.StatusCode = 404;
                // En .NET Core, on écrit sur le body de manière asynchrone idéalement, 
                // mais dans une méthode void, on écrit directement.
                // Hm.Write gère l'écriture safe dans le flux de sortie.
                this.Hm.Write("Fichier non trouvé ou accès interdit.");
                return;
            }

            // Configuration de la réponse pour le téléchargement
            response.Clear();
            response.ContentType = "application/octet-stream"; // Ou type spécifique
            
            // Syntax Headers.Append en .NET Core
            response.Headers.Append("Content-Disposition", "attachment; filename=\"" + fileName + "\"");

            // Envoi du fichier
            try 
            {
                response.SendFileAsync(filePath).GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                Sys.Log("Erreur téléchargement : " + ex.Message);
            }
        }
```


```
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Sinequa.Configuration;
using Sinequa.Plugins;

namespace Sinequa.Plugin
{
    public class MyDownloadPlugin : HttpEndpointBuilderPlugin
    {
        public override void Configure(IApplicationBuilder app)
        {
            // Définit l'URL d'accès : /api/custom/download
            app.Map("/api/custom/download", HandleDownload);
        }

        private static async void HandleDownload(IApplicationBuilder app)
        {
            app.Run(async context =>
            {
                // 1. Récupération du nom de fichier
                string fileName = context.Request.Query["file"];
                
                // 2. Sécurisation du chemin (A adapter !)
                string baseDir = @"C:\sinequa\data\mon_dossier_partage\"; 
                string filePath = Path.Combine(baseDir, fileName);

                // Vérification basique de sécurité
                if (File.Exists(filePath) && Path.GetFullPath(filePath).StartsWith(baseDir))
                {
                    // 3. Envoi du fichier
                    context.Response.ContentType = "application/octet-stream";
                    context.Response.Headers.Append("Content-Disposition", $"attachment; filename=\"{fileName}\"");
                    await context.Response.SendFileAsync(filePath);
                }
                else
                {
                    context.Response.StatusCode = 404;
                    await context.Response.WriteAsync("Fichier non trouvé.");
                }
            });
        }
    }
}
```



```
using System;
using System.IO;
using Sinequa.Common;
using Sinequa.Plugins;

namespace Sinequa.Plugin
{
    public class DownloadFilePlugin : JsonMethodPlugin
    {
        // 1. Configuration du format de sortie (Indispensable)
        // D'après la source [2], cela indique que la sortie est un Blob (binaire)
        public override bool OnInitJsonFormat()
        {
            Method.MethodFormat = JsonMethodFormat.Post_Json_To_Blob;
            return true;
        }

        // 2. Gestion des droits (Source [3])
        // Permettre aux utilisateurs (pas seulement admins) de télécharger
        public override JsonMethodAuthLevel GetRequiredAuthLevel()
        {
            return JsonMethodAuthLevel.User;
        }

        // 3. Logique principale
        public override void OnPluginMethod()
        {
            // Récupérer le nom du fichier depuis l'URL (ex: &file=mondoc.pdf)
            // Note : En GET, les paramètres sont dans JsonRequest (Source [4])
            string fileName = this.JsonRequest.ValueStr("file");
            
            // ATTENTION : Sécurisez ce chemin pour éviter les failles de sécurité !
            string baseDir = @"C:\sinequa\data\docs\"; 
            string filePath = Path.Combine(baseDir, fileName);

            if (File.Exists(filePath))
            {
                // Lecture du fichier en octets
                byte[] fileBytes = File.ReadAllBytes(filePath);

                // D'après la source [2] :
                // On assigne le contenu binaire à l'objet BlobResponse géré par Sinequa
                Method.BlobResponse = new Blob(fileBytes);
                
                // On définit le nom qui apparaîtra lors du téléchargement
                Method.BlobResponseFilename = fileName; 
                
                // On force le navigateur à télécharger le fichier (pas d'affichage dans l'onglet)
                Method.BlobResponseForceDownload = true; 
            }
            else
            {
                Sys.Log("Fichier introuvable : " + filePath);
                // En cas d'erreur, le retour sera vide ou une erreur HTTP générique
            }
        }
    }
}
```