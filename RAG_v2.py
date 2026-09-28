import argparse
import ollama
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# ---------------------- CONFIG ---------------------- #
load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT")
}

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3.2"


# ---------------------- DATABASE ---------------------- #
def connect_db():
    """สร้าง connection ไปยัง PostgreSQL database"""
    return psycopg2.connect(**DB_CONFIG)


def query_postgresql(question_embedding_str, k=5):
    """
    ค้นหาเอกสารที่ใกล้เคียงที่สุดจาก embeddings ใน PostgreSQL
    """
    sql_query = """
        SELECT content, embedding <=> %s::vector AS similarity_score
        FROM documents
        ORDER BY similarity_score ASC
        LIMIT %s
    """
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(sql_query, (question_embedding_str, k))
        return cur.fetchall()


def add_embedding(content, embedding):
    """เพิ่มเอกสารและ embedding ลงใน PostgreSQL"""
    insert_sql = "INSERT INTO documents (content, embedding) VALUES (%s, %s)"
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(insert_sql, (content, embedding))
        conn.commit()


# ---------------------- EMBEDDING ---------------------- #
def txt_embedding(text):
    """แปลงข้อความเป็น embedding vector"""
    embedder = SentenceTransformer(EMBED_MODEL)
    return embedder.encode(text).tolist()


# ---------------------- LLM ---------------------- #
def generate_response(context, query_text):
    """
    สร้างคำตอบจาก LLM โดยใช้ context ที่ retrieve มา
    """
    prompt = f"""
    Answer the question based on the following context:
    Context: {context}
    Question: {query_text}
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


# ---------------------- MAIN TASKS ---------------------- #
def run_rag():
    """โหมด RAG: รับคำถาม → หา embedding → query DB → ส่งเข้า LLM"""
    question = "ขาแมวกับขาหมูรวมกันได้เท่าไหร่"
    print(f"\n❓ คำถาม: {question}\n")

    # 1) แปลงคำถามเป็น embedding
    question_embedding = txt_embedding(question)
    question_embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"

    # 2) query เอกสารที่ใกล้เคียง
    results = query_postgresql(question_embedding_str, k=5)
    print("🔎 Query Results:", results)

    # 3) รวม context
    context = ",".join(content for content, _ in results)
    print("\n📄 Context:", context)

    # 4) สร้างคำตอบ
    answer = generate_response(context, question)
    print("\n💡 คำตอบ:", answer)


def run_add_embeddings():
    """โหมด Add Embedding: เพิ่มเอกสารใหม่เข้า DB"""
    documents = [
        "หมูมี 4 ขา",
        "วัวมี 4 ขา",
        "นกมี 2 ขา"
    ]
    for doc in documents:
        embedding = txt_embedding(doc)
        add_embedding(doc, embedding)
        print(f"✅ Added: {doc}")


# ---------------------- ENTRY POINT ---------------------- #
def main():
    parser = argparse.ArgumentParser(description="RAG + Embedding System")
    parser.add_argument(
        "mode",
        choices=["rag", "add"],
        help="เลือกโหมดการทำงาน: rag = ถามตอบด้วย RAG, add = เพิ่ม embeddings เข้า DB"
    )
    args = parser.parse_args()

    if args.mode == "rag":
        run_rag()
    elif args.mode == "add":
        run_add_embeddings()


if __name__ == "__main__":
    main()
    
# =========== How to run ============
# python file_name.py {select add/rag}
# example: python RAG_v2.py add