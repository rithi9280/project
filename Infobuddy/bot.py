from flask import Flask, request, jsonify
import PyPDF2
import openai
import faiss
import numpy as np

app = Flask(__name__)
openai.api_key = "your_openai_api_key"

# Create a FAISS index
dimension = 1536  # Embedding vector size for `text-embedding-ada-002`
index = faiss.IndexFlatL2(dimension)
pdf_texts = []  # Store the text data separately

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_content = ""

    for page in pdf_reader.pages:
        pdf_content += page.extract_text()
    
    pdf_texts.append(pdf_content)

    # Embed and add to FAISS
    embedding = get_embedding(pdf_content)
    index.add(np.array([embedding]))

    return jsonify({"message": "PDF uploaded and indexed successfully!"})

@app.route('/query', methods=['POST'])
def query_pdf():
    data = request.json
    query = data['query']

    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Search FAISS for the most relevant text
    distances, indices = index.search(np.array([query_embedding]), k=1)
    best_match_idx = indices[0][0]
    best_match_text = pdf_texts[best_match_idx]

    # Generate response using GPT
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the document: {best_match_text}, answer this: {query}"}
        ]
    )

    return jsonify({"response": response.choices[0].message['content']})

def get_embedding(text):
    """Get text embedding using OpenAI's embedding model."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

if __name__ == '__main__':
    app.run(debug=True)
