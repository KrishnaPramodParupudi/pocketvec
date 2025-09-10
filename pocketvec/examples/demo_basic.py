
#   ***************************************************
#   Follow this demo if you want to create embeddings,
#   retrieve relevant data chunks in one go at a time. 
#   ***************************************************


from pocketvec import PocketVec

HF_TOKEN = "YOUR_HF_TOKEN"

vecstore = PocketVec(hf_token=HF_TOKEN, model_name="all-MiniLM-L6-v2")

texts = ["I love machine learning", "Python is great for AI", "I enjoy NLP tasks"]
vecstore.generate_embeddings(texts)

result = vecstore.query_similar(
    "Tell me about AI", 
    similarity_type="cosine", 
    no_of_results=2
    )

print(result)