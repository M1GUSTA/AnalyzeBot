import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
gpt_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  

contract_text = """
1. The party agrees to deliver the goods within 30 days.
2. However, in Section 4, the delivery timeframe is extended to 45 days under special circumstances.
3. The seller must notify the buyer about any delivery delays at least 5 days in advance.
4. Failure to deliver within the agreed time results in penalties.
"""

def analyze_contract(text):
    paragraphs = text.strip().split("\n")
    
    embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)
    
    for i, paragraph in enumerate(paragraphs):
        print(f"\nАнализ абзаца {i+1}: {paragraph}")
        for j in range(i+1, len(paragraphs)):
            similarity = util.cos_sim(embeddings[i], embeddings[j])
            if similarity > 0.5:  
                print(f"⚠️ Потенциальное совпадение или противоречие с абзацем {j+1}: {paragraphs[j]}")
    
    prompt = f"Review this legal contract for contradictions: {text}. Highlight problems and give suggestions."
    recommendation = gpt_model(prompt, max_length=150, num_return_sequences=1)
    
    print("\nРекомендации ИИ:")
    print(recommendation[0]["generated_text"])

analyze_contract(contract_text)
