from flask import Flask, request, jsonify
import openai
import numpy as np
import faiss

app = Flask(__name__)
openai.api_key = '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'



# Load embeddings into FAISS
dimension = 1536
index = faiss.IndexFlatL2(dimension)
texts = []  # Store the corresponding text data

@app.route('/upload', methods=['POST'])
def upload_file():
    # Extract and vectorize text here...
    texts.append("Extracted text example")
    embedding = get_embeddings("Extracted text example")
    index.add(np.array([embedding]))
    return jsonify({"message": "File uploaded and indexed."})

@app.route('/query', methods=['POST'])
def query():
    query = request.json['query']
    query_embedding = get_embeddings(query)
    distances, indices = index.search(np.array([query_embedding]), k=1)
    response = texts[indices[0][0]]

    # Generate an AI-based response
    ai_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on this text: {response}, answer the query: {query}"}
        ]
    )
    return jsonify({"response": ai_response['choices'][0]['message']['content']})

if __name__ == '__main__':
    app.run(debug=True)
