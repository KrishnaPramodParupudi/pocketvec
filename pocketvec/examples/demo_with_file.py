
#   **************************************************************
#   Follow this demo if you want to create embeddings, sotre them
#   retrieve relevant data chunks at a later point of time. 
#   **************************************************************


from pocketvec import PocketVec

HF_TOKEN = "YOUR_HF_TOKEN"

vecstore = PocketVec(hf_token=HF_TOKEN, model_name="all-MiniLM-L6-v2")

text = ["I love machine learning", "Python is great for AI", "I enjoy NLP tasks"]
vecstore.generate_embeddings(text)
vecstore.save_embeddings("embedtest.txt")

result = vecstore.query_similar(
    "Tell me about AI", 
    similarity_type="cosine", 
    no_of_results=2, 
    file_path="Embeddings/embedtest.txt",
    texts=text
    )
print(result)