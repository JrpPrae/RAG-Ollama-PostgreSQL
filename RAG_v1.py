import ollama
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()


def generate_response(context, query_text):
    
    prompt = f"""Answer the question based on following
    context: {context}
    Question: {query_text}
    """
    
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role":"system", "content":"You are an assistant."},
            {"role":"user", "content": prompt}
        ]
    )
    
    return response["message"]["content"]

def connect_db():
    conn = psycopg2.connect(
        dbname=os.getenv("MYDB"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT")
    )
    
    return conn
    
def txt_embedding(text):
    embedder = SentenceTransformer("BAAI/bge-m3")
    embedding = embedder.encode(text).tolist()
    
    return embedding

def query_postgresql(question_embedding_str, k=5):
    conn = connect_db()
    cur = conn.cursor()
    sql_query = """
        SELECT content, embedding <=> %s::vector AS similarity_score
        FROM documents
        ORDER BY similarity_score ASC
        LIMIT %s
    """
    cur.execute(sql_query, (question_embedding_str, k))
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result
    
def add_embedding(content, embedding):
    conn = connect_db()
    cur = conn.cursor()
    
    insert_sql = """INSERT INTO documents (content, embedding) VALUES (%s, %s)"""
    cur.execute(insert_sql, (content, embedding))
    conn.commit()
    
    cur.close()
    conn.close()

def main():
    select_status = "2"
    status = {"1":"Add Embedding", "2":"RAG"}
    
    if select_status in status.keys():
        if select_status == "2":
    
            question = "ขาแมวกับขาหมูรวมกันได้เท่าไหร่"
            print(question + "\n")
            
            question_embedding = txt_embedding(question)
            # print(question_embedding[0])
            
            question_embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"
            # print(question_embedding_str)
            
            query_result = query_postgresql(question_embedding_str, 5)
            print(query_result) # return [(content, similarity_score)]
            
            context = ",".join(content[0] for content in query_result)
            print(context)
            
            response_llm = generate_response(question, context)
            print("\nคำตอบ: " + response_llm)
            
        else:
            documents = [
                "หมูมี 4 ขา",
                "วัวมี 4 ขา",
                "นกมี 2 ขา"
            ]

            for doc in documents:
                embedding = txt_embedding(doc)
                add_embedding(doc, embedding)
    
    
if __name__ == "__main__":
    main()