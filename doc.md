Structure attendue :
{
    "message_chat": "Texte de réponse pour l'utilisateur...",
    "doc_type": "pdf" | "docx" | "xlsx" | "pptx" | null,
    "doc_data": {
        "titre": "Titre du document",
        "contenu_texte": "Paragraphes pour Word/PDF...", 
        "liste_points": ["Point 1", "Point 2"], (uniquement pour PPTX)
        "donnees_tabulaires": [{"Col1": val, "Col2": val}] (uniquement pour Excel)
    }
}

Si l'utilisateur ne demande pas de document, mets "doc_type": null.