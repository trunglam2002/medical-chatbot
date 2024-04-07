from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeStore
import pinecone
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from src.prompt import *
import os
import torch

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "vni-medical"

docsearch = PineconeStore.from_existing_index(index_name, embeddings)

model_path = "vinai/PhoGPT-4B-Chat"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.init_device = "cuda"

load_dir = 'model/vietnamese7b-llama/models--vinai--PhoGPT-4B-Chat/snapshots/116013fa63f8c4025739487e1cbff65b7375bbe2'
model = AutoModelForCausalLM.from_pretrained(
    load_dir, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)


def clean_qa(input, docsearch, model):
    similar = docsearch.similarity_search(input, k=1)
    instruction = f"Sử dụng thông tin sau và kiến thức của bạn: {similar}. Hãy trả lời câu hỏi: {input}"

    input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
    input_ids = tokenizer(input_prompt, return_tensors="pt")

    print("Enter generate")
    outputs = model.generate(
        inputs=input_ids["input_ids"].to("cuda"),
        attention_mask=input_ids["attention_mask"].to("cuda"),
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = response.split("### Trả lời:")[1]
    print("End generate")
    return response


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input_text = msg
    print(input_text)
    result = clean_qa(input_text, docsearch, model)
    print('Response : ', result)
    return jsonify(response=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
