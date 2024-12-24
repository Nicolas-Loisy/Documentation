# IA et RAG

## Ressources

* Cours Udemy sur ChatGPT et LangChain: [https://www.udemy.com/course/chatgpt-and-langchain-the-complete-developers-masterclass/](https://www.udemy.com/course/chatgpt-and-langchain-the-complete-developers-masterclass/)
* Discord LangChain: [https://discord.gg/vvcyvjDkdC](https://discord.gg/vvcyvjDkdC)

## LLMs Gratuit et Local

* Intégrateur/Chat multi-modèles: [https://lmstudio.ai/](https://lmstudio.ai/)
* Gestionnaire/Observer de LLM: Langfuse
* Anglais: TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Embeddings

* Français: antoinelouis/biencoder-electra-base-french-mmarcoFR
* Anglais: sentence-transformers/all-mpnet-base-v2
* Multilingue: sentence-transformers/paraphrase-multilingual-mpnet-base-v2

## Autres ressources

* [https://swharden.com/blog/2023-07-30-ai-document-qa/](https://swharden.com/blog/2023-07-30-ai-document-qa/)
* [https://replicate.com/blog/run-llama-locally](https://replicate.com/blog/run-llama-locally)
* [https://swharden.com/blog/2023-07-29-ai-chat-locally-with-python/](https://swharden.com/blog/2023-07-29-ai-chat-locally-with-python/)
* [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf?library=true](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf?library=true)

## Définitions

## Apprentissage automatique

* **LLMs (Large Language Model):**
    * Exemples: GPT-3, Bard
    * Modèles de langage de grande taille
    * Formés sur un ensemble de données massif
    * Générer du texte, traduire des langues, écrire du contenu créatif, répondre à des questions

* **Chain (Chaîne dans LangChain):**
    * Séquence d'étapes pour traiter une requête
    * Chaque étape est un composant (modèle de langage, retriever, stratégie de recherche)
    * Types de chaînes:
        * `refine`: Affiner les résultats d'une chaîne précédente
        * `map_rerank`: Combiner les résultats de plusieurs chaînes
        * `map_reduce`: Créer un résumé par document
        * `stuff`: Intégrer des informations contextuelles d'un entrepôt de vecteurs

* **Langchain:**
    * Framework open source pour développer des applications alimentées par des modèles de langage
    * Permet de créer des applications contextuellement conscientes
    * Comprendre et répondre aux questions et requêtes des utilisateurs

## Représentation du langage

* **Embeddings:**
    * Représentation d'un mot ou d'une phrase sous forme d'un vecteur
    * Transformer un texte en une liste de nombres

* **Embeddings Redundant Filter:**
    * Filtrer les documents redondants dans un ensemble de documents
    * Comparer les embeddings des documents

## Recherche

* **Retrieval:**
    * Rechercher des documents pertinents dans un corpus de texte

* **Retrievers:**
    * Composants responsables de la récupération de documents pertinents

* **Similarity search:**
    * Rechercher dans un VectorStore les documents dont les vecteurs sont les plus proches de celui de la requête

## Données

* **Dataset:**
    * Collection de données pour entraîner un modèle d'apprentissage automatique

* **Document_loaders:**
    * Chargement de documents à partir de diverses sources

* **Document_transformers:**
    * Prétraitement et nettoyage des documents

## Sortie

* **Output:**
    * Résultat final d'un modèle d'apprentissage automatique

## Analyse et Prétraitement

* **Parsers:**
    * Un parser structure et interprète les contenus textuelles en formats utilisables (ex: Récupérer les textes d’un PDF).
    * Syntaxe du texte dans un format structuré, tel qu'un arbre ou un graphe.

* **Document Transformer:**
    * Convertir un document en un autre (ex: HTML en texte)

* **Document splitter:**
    * Découper un document en plusieurs sous-documents

* **Text_splitter:**
    * Diviser le texte en plus petits morceaux

* **Text_transformers:**
    * Transformer le texte pour l'entrée d'un modèle d'apprentissage automatique

## Mémoire

* **Memory:**
    * Stockage et récupération d'informations

## Stockage

* **Vectorstores:**
    * Bases de données stockant le texte, les métadonnées et les vecteurs

* **Chromadb:**
    * Base de données NoSQL pour stocker et rechercher des données linguistiques

* **Chroma:**
    * Kit d'analyse de texte pour extraire

* **Mongodb:**
    * Base de données NoSQL utilisée pour stocker et récupérer de grandes quantités de données non structurées.

* **Solr:**
    * Un moteur de recherche qui est utilisé pour rechercher des documents texte.

## AgentExecutor

- **LangChain - AgentExecutor:** Classe permettant d'exécuter un agent.
- **Agent:** Classe utilisant un modèle de langage pour effectuer une tâche, répondre à une question ou résoudre un problème.

**1. Embeddings**

* **Création d'embeddings:**

    * Un modèle d'apprentissage automatique (encodeur, souvent un réseau neuronal) est entraîné sur une grande quantité de texte. Le modèle apprend à associer des mots et des phrases à des vecteurs de nombres d'une dimension spécifique (généralement entre 768 et 1536). Ces vecteurs capturent les relations sémantiques et contextuelles entre les mots.

* **Détermination des nombres des vecteurs:**
    * Les nombres du vecteur sont des valeurs continues (généralement entre -1 et 1) calculées par le modèle d'apprentissage automatique pendant l'entraînement. Ces valeurs n'ont pas de signification individuelle et ne peuvent être interprétées directement.

* **Signification des nombres:**
    * La signification réside dans la position relative des nombres dans le vecteur et dans la distance entre les vecteurs de différents mots ou phrases. Des vecteurs proches dans l'espace vectoriel représentent des concepts similaires ou sémantiquement liés.

**2. Génération de la réponse finale**

* **Préparation de la réponse:**
    * La question et les documents pertinents sont transformés en une représentation adaptée au modèle de langage utilisé pour la génération de texte. Cette représentation peut inclure des embeddings des mots et des phrases clés.

* **Génération de texte:**
    * Le modèle de langage de génération de texte reçoit la représentation préparée et génère une réponse candidate.

* **Raffinement (facultatif):**
    * La réponse candidate peut être affinée en utilisant des techniques comme la paraphrase ou la traduction pour améliorer la clarté et la fluidité.
* **Sortie:**
    * La réponse finale est fournie à l'utilisateur.