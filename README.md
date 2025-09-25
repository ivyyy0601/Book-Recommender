# 📚 Semantic Book Recommender

This project builds an **end-to-end pipeline** for analyzing book descriptions, extracting emotions, and recommending books using **LLMs (Large Language Models)**, **vector search**, and an **interactive Gradio dashboard**.

---
## 🛠 Tech Stack
- **Python** (pandas, numpy)  
- **Hugging Face Transformers** (classification & emotion analysis)  
- **LangChain + Chroma** (vector search)  
- **Gradio** (interactive UI)  

---

## ⭐️ Chapters1: data exploration:  data-exploration.ipynb ⭐️

### 1. Introduction to getting and preparing text data  
- Load book metadata from CSV  
- Explore text fields such as `title`, `subtitle`, and `description`  

### 2. Starting a new PyCharm project  
- Setup Python environment  
- Install dependencies (`transformers`, `langchain`, `gradio`, etc.)  

### 3. Patterns of missing data  
- Identify missing values in `subtitle` and `description`  
- Decide how to handle them  

### 4. Checking the number of categories  
- Analyze book categories  
- Group into simplified categories for easier recommendations  

### 5. Remove short descriptions  
- Drop rows with very short or uninformative descriptions  

### 6. Final cleaning steps  
- Merge `title + subtitle`  
- Create `tagged_description` field for embeddings  

---

## ⭐️  Chapters2: Working with LLMs and Vector Search   vector-search.ipynb  ⭐️ 

### 7. Introduction to LLMs and vector search  
- Concept of embeddings for semantic similarity  

### 8. LangChain  
- Use `LangChain` to handle embeddings and vector DB integration  

### 9. Splitting the books using `CharacterTextSplitter`  
- Prepare text for embeddings  
- Split into chunks for indexing  

### 10. Building the vector database  
- Store embeddings in `ChromaDB`  
- Enable semantic similarity search  

### 11. Getting book recommendations using vector search  
- Query embeddings → retrieve similar books  

---

## ⭐️  Chapters3: Text Classification and Emotions  text-classification.ipynb   ⭐️ 

### 12. Introduction to zero-shot text classification using LLMs  
- Classify book genres or themes without labeled data  

### 13. Finding LLMs for zero-shot classification on Hugging Face  
- Example: `facebook/bart-large-mnli`  

### 14. Classifying book descriptions  
- Apply classifier to each description  

### 15. Checking classifier accuracy  
- Evaluate results against known categories  

## ⭐️  Chapter4: Emotions Classification   sentiment-analysis.ipynb  ⭐️ 

### 16. Introduction to using LLMs for sentiment analysis  
- Extract emotional tone from descriptions  

### 17. Finding fine-tuned LLMs for sentiment analysis  
- Example: `j-hartmann/emotion-english-distilroberta-base`  

### 18. Extracting emotions from book descriptions  
- Run model per sentence  
- Aggregate max scores for each emotion (`joy`, `fear`, `anger`, etc.)  
- Save results into `books_with_emotions.csv`  

---

## 🎨 Building the Dashboard  

### 19. Introduction to Gradio  
- Simple UI framework for ML demos  

### 20. Building a Gradio dashboard to recommend books  
- User enters query + selects category/tone  
- System retrieves recommendations using embeddings + emotion scores  
- Display results in `Gallery` with thumbnails  





