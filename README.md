# RAG with Ollama + PostgreSQL (pgvector)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-000000?logo=ollama&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Embedding](https://img.shields.io/badge/Embedding-BAAI%2Fbge--m3-FFD21E?logo=huggingface&logoColor=black)

ระบบ **RAG (Retrieval-Augmented Generation)** ที่ **รันบนเครื่องทั้งหมด (local)** ไม่ต้องใช้ API ภายนอก ใช้

- **Ollama** รัน LLM `llama3.2`
- **PostgreSQL + pgvector** เป็น Vector Database
- **BAAI/bge-m3** (Sentence Transformers) แปลงข้อความเป็น embedding ที่รองรับหลายภาษารวมถึงภาษาไทย

ตัวอย่างในโปรเจกต์คือการเก็บข้อเท็จจริงสั้น ๆ (เช่น "แมวมี 4 ขา", "คนมี 2 ขา") ลงฐานข้อมูล แล้วถามคำถามที่ต้องใช้ข้อมูลเหล่านั้น เช่น "ขาแมวกับขาคนรวมกันได้เท่าไหร่" ระบบจะค้นเอกสารที่เกี่ยวข้องจาก PostgreSQL แล้วส่งให้ LLM ตอบ

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [Workflow](#workflow)
- [Vector Search ใน PostgreSQL](#vector-search-ใน-postgresql)
- [Tech Stack](#tech-stack)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [Environment Variables](#environment-variables)
- [วิธีติดตั้งและใช้งาน](#วิธีติดตั้งและใช้งาน)
- [ตัวอย่างผลลัพธ์](#ตัวอย่างผลลัพธ์)
- [หมายเหตุและสิ่งที่ควรพัฒนาต่อ](#หมายเหตุและสิ่งที่ควรพัฒนาต่อ)

## ภาพรวม

RAG ช่วยให้ LLM ตอบคำถามโดยอ้างอิงข้อมูลที่เราเตรียมไว้ แทนการตอบจากความรู้ที่โมเดลจำมาเอง ลดปัญหาการแต่งคำตอบ (hallucination) และใช้กับข้อมูลเฉพาะขององค์กรได้ โปรเจกต์นี้แบ่งการทำงานเป็น 2 โหมด:

| โหมด | หน้าที่ |
| --- | --- |
| **add** (Indexing) | แปลงเอกสารเป็น embedding แล้วบันทึกลงตาราง `documents` ใน PostgreSQL |
| **rag** (Query) | แปลงคำถามเป็น embedding → ค้นเอกสารที่ใกล้เคียงที่สุด → ส่งเป็น context ให้ LLM ตอบ |

ไฟล์ในโปรเจกต์:

| ไฟล์ | รายละเอียด |
| --- | --- |
| [`rag_test.ipynb`](rag_test.ipynb) | Notebook ทดลองทีละขั้น: เรียก Ollama, เชื่อม PostgreSQL, เปิด extension `vector`, สร้างตาราง, embedding, query และ generate คำตอบ |
| [`RAG_v1.py`](RAG_v1.py) | รวมโค้ดจาก notebook เป็นสคริปต์ เลือกโหมดด้วยตัวแปร `select_status` ในโค้ด |
| [`RAG_v2.py`](RAG_v2.py) | เวอร์ชันปรับปรุง: แยก config, มี docstring, ใช้ context manager และเลือกโหมดผ่าน **command line (`argparse`)** — **แนะนำให้ใช้ไฟล์นี้** |

## Workflow

### ภาพรวม RAG Pipeline

```text
┌──────────────────────────── Indexing  (python RAG_v2.py add) ────────────────────────────┐
│                                                                                          │
│  เอกสาร ["หมูมี 4 ขา", "วัวมี 4 ขา", "นกมี 2 ขา"]                                            │
│        │                                                                                 │
│        ▼  txt_embedding()                                                                │
│  SentenceTransformer("BAAI/bge-m3")  ──▶  vector 1024 มิติ                                  │
│        │                                                                                 │
│        ▼  add_embedding()                                                                │
│  INSERT INTO documents (content, embedding) VALUES (%s, %s)                              │
│                                                                                          │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           ▼
                        ┌────────────────────────────────────┐
                        │  PostgreSQL + pgvector (Docker)    │
                        │  documents(id, content,            │
                        │            embedding vector(1024)) │
                        └────────────────────────────────────┘
                                           ▲
┌──────────────────────────── Query  (python RAG_v2.py rag) ──┴────────────────────────────┐
│                                                                                          │
│  คำถาม "ขาแมวกับขาหมูรวมกันได้เท่าไหร่"                                                       │
│        │                                                                                 │
│        ▼  1) txt_embedding(question) → "[0.01, -0.05, ...]"                              │
│        │                                                                                 │
│        ▼  2) query_postgresql(k=5)                                                       │
│  SELECT content, embedding <=> %s::vector AS similarity_score                            │
│  FROM documents ORDER BY similarity_score ASC LIMIT 5                                    │
│        │                                                                                 │
│        ▼  3) รวม content เป็น context: "หมูมี 4 ขา,วัวมี 4 ขา,..."                            │
│        │                                                                                 │
│        ▼  4) generate_response(context, question)                                        │
│  ollama.chat(model="llama3.2", messages=[system, user(prompt)])                          │
│        │                                                                                 │
│        ▼                                                                                 │
│  คำตอบภาษาไทยที่อ้างอิงจาก context                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Prompt ที่ส่งให้ LLM

```text
system: You are an assistant.
user:   Answer the question based on the following context:
        Context: {context}
        Question: {query_text}
```

### ขั้นตอนใน Notebook (`rag_test.ipynb`)

1. ทดสอบเรียก `llama3.2` ผ่าน `ollama.chat()`
2. เชื่อมต่อ PostgreSQL ด้วย `psycopg2` (container ต้องรันอยู่)
3. เปิด extension `CREATE EXTENSION vector` และตรวจสอบใน DBeaver
4. สร้างตาราง `documents` ที่มีคอลัมน์ `embedding vector(1024)`
5. ทดสอบ embedding ด้วย `bge-m3` (ได้ vector ยาว 1024)
6. เพิ่มเอกสารตัวอย่างลงฐานข้อมูล
7. Query เอกสารที่ใกล้เคียงกับคำถาม
8. Generate คำตอบด้วย LLM จาก context ที่ค้นได้

## Vector Search ใน PostgreSQL

**pgvector** เป็น extension ที่เพิ่มชนิดข้อมูล `vector` และตัวดำเนินการสำหรับวัดระยะห่างระหว่าง vector ให้ PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT,
    embedding vector(1024)      -- ต้องตรงกับขนาด output ของ bge-m3
);
```

| ตัวดำเนินการ | ความหมาย |
| --- | --- |
| `<=>` | **Cosine distance** (ใช้ในโปรเจกต์นี้) — ค่า **น้อย = ใกล้เคียงมาก** จึงเรียง `ASC` |
| `<->` | Euclidean (L2) distance |
| `<#>` | Negative inner product |

รายละเอียดของ query:

- `embedding <=> %s::vector` คำนวณ cosine distance ระหว่าง embedding ในตารางกับ embedding ของคำถาม
- embedding ของคำถามถูกแปลงเป็น string รูปแบบ `"[x1,x2,...]"` แล้ว cast เป็น `vector`
- `ORDER BY similarity_score ASC LIMIT k` เลือก `k` เอกสารที่ใกล้เคียงที่สุด

## Tech Stack

| หมวด | เทคโนโลยี | หน้าที่ในโปรเจกต์ |
| --- | --- | --- |
| ภาษา | **Python 3.12** | ภาษาหลัก |
| LLM Runtime | **[Ollama](https://ollama.com/)** + Python client `ollama` | รัน LLM บนเครื่อง |
| LLM | **Llama 3.2** (`llama3.2`) | สร้างคำตอบจาก context |
| Embedding Model | **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)** ผ่าน `sentence-transformers` | แปลงข้อความ (รองรับภาษาไทย) เป็น vector 1024 มิติ |
| Vector Database | **PostgreSQL** + **[pgvector](https://github.com/pgvector/pgvector)** | เก็บ embedding และค้นหาด้วย cosine distance |
| DB Driver | **psycopg2** | เชื่อมต่อและรันคำสั่ง SQL จาก Python |
| Container | **Docker** (image `pgvector/pgvector`) | รัน PostgreSQL ที่ติดตั้ง pgvector แล้ว |
| DB Management | **DBeaver** | ดูตาราง ตรวจสอบ extension และข้อมูล |
| Configuration | **python-dotenv** | โหลดค่าเชื่อมต่อฐานข้อมูลจาก `.env` |
| CLI | **argparse** | เลือกโหมด `add` / `rag` ใน `RAG_v2.py` |
| Notebook | **Jupyter** | ทดลองทีละขั้นใน `rag_test.ipynb` |

## โครงสร้างโปรเจกต์

```text
RAG-Ollama-PostgreSQL/
├── rag_test.ipynb     # Notebook ทดลอง RAG ทีละขั้น (รวมคำสั่งสร้าง extension และตาราง)
├── RAG_v1.py          # สคริปต์เวอร์ชันแรก (เลือกโหมดด้วยตัวแปรในโค้ด)
├── RAG_v2.py          # สคริปต์เวอร์ชันปรับปรุง (CLI: add / rag)
├── .env               # (ต้องสร้างเอง) ค่าเชื่อมต่อฐานข้อมูล
├── .gitignore         # กัน *.env และ docs.txt
└── README.md
```

### ฟังก์ชันใน `RAG_v2.py`

| ฟังก์ชัน | หน้าที่ |
| --- | --- |
| `connect_db()` | เชื่อมต่อ PostgreSQL ตาม `DB_CONFIG` |
| `txt_embedding(text)` | แปลงข้อความเป็น embedding (list ของ float) ด้วย `bge-m3` |
| `add_embedding(content, embedding)` | `INSERT` เอกสารและ embedding ลงตาราง `documents` |
| `query_postgresql(question_embedding_str, k)` | ค้นเอกสารที่ใกล้เคียงที่สุด `k` รายการ คืนค่า `[(content, distance), ...]` |
| `generate_response(context, query_text)` | สร้าง prompt และเรียก `llama3.2` ผ่าน Ollama |
| `run_add_embeddings()` | โหมด `add`: เพิ่มเอกสารตัวอย่างลงฐานข้อมูล |
| `run_rag()` | โหมด `rag`: embedding คำถาม → query → รวม context → ถาม LLM |

## Environment Variables

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์ (ถูกกันไว้ใน [`.gitignore`](.gitignore) แล้ว):

```env
DBNAME=mydb
USER=admin
PASSWORD=your_password
HOST=localhost
PORT=5432
```

| ตัวแปร | ใช้ใน | คำอธิบาย |
| --- | --- | --- |
| `DBNAME` | `RAG_v2.py` | ชื่อฐานข้อมูล |
| `MYDB` | `RAG_v1.py` | ชื่อฐานข้อมูล (v1 ใช้ชื่อตัวแปรนี้แทน `DBNAME`) |
| `USER` | ทั้งสองไฟล์ | ชื่อผู้ใช้ PostgreSQL |
| `PASSWORD` | ทั้งสองไฟล์ | รหัสผ่าน PostgreSQL |
| `HOST` | ทั้งสองไฟล์ | ที่อยู่ฐานข้อมูล (เช่น `localhost`) |
| `PORT` | ทั้งสองไฟล์ | พอร์ต (ค่าเริ่มต้น `5432`) |

> บน macOS / Linux ระบบตั้งตัวแปร `USER` (ชื่อผู้ใช้ของเครื่อง) ไว้อยู่แล้ว และ `load_dotenv()` จะไม่เขียนทับค่าที่มีอยู่ ดูวิธีแก้ใน [หมายเหตุ](#หมายเหตุและสิ่งที่ควรพัฒนาต่อ)

## วิธีติดตั้งและใช้งาน

### 1. ติดตั้ง Ollama และดาวน์โหลดโมเดล

ดาวน์โหลด Ollama จาก <https://ollama.com/download> แล้วรัน:

```bash
ollama pull llama3.2
```

### 2. รัน PostgreSQL + pgvector ด้วย Docker

```bash
docker run -d --name pgvector \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 3. เปิด extension และสร้างตาราง

เชื่อมต่อด้วย DBeaver หรือ `psql` แล้วรัน:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT,
    embedding vector(1024)
);

-- ตรวจสอบว่าเปิด extension แล้ว
SELECT * FROM pg_catalog.pg_extension WHERE extname = 'vector';
```

(หรือเอา comment ออกจาก cell ที่เกี่ยวข้องใน `rag_test.ipynb` แล้วรัน)

### 4. ติดตั้ง Python dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install ollama psycopg2-binary sentence-transformers python-dotenv jupyter
```

> ครั้งแรกที่รันจะดาวน์โหลดโมเดล `BAAI/bge-m3` จาก Hugging Face (ขนาดประมาณ 2 GB)

### 5. สร้างไฟล์ `.env`

ดูหัวข้อ [Environment Variables](#environment-variables)

### 6. รัน

```bash
# เพิ่มเอกสารตัวอย่างลงฐานข้อมูล
python RAG_v2.py add

# ถามคำถามด้วย RAG
python RAG_v2.py rag
```

คำถามและเอกสารตัวอย่างกำหนดไว้ในฟังก์ชัน `run_rag()` และ `run_add_embeddings()` แก้ได้ตามต้องการ

## ตัวอย่างผลลัพธ์

จาก `rag_test.ipynb` (เอกสาร: "หมามี 4 ขา", "คนมี 2 ขา", "แมวมี 4 ขา"):

```text
คำถาม: ขาแมวกับขาคนรวมกันได้เท่าไหร่

ผลการค้นหา (content, cosine distance):
  ('แมวมี 4 ขา', 0.2625)
  ('คนมี 2 ขา', 0.3394)
  ('หมามี 4 ขา', 0.3519)

คำตอบจาก llama3.2:
  ตามที่กล่าวไว้แล้ว
  * แมว มีขา 4 ขา
  * คน มีขา 2 ขา
  ดังนั้น หากเราบวกขาของแมวและคนเข้าด้วยกัน จะได้รวมขา 6 ขา (4 ขาของแมว + 2 ขาของคน)
```

เอกสารที่เกี่ยวข้องที่สุด ("แมว" และ "คน") ได้ distance ต่ำที่สุด และ LLM ใช้ข้อมูลจาก context คำนวณคำตอบได้ถูกต้อง

## หมายเหตุและสิ่งที่ควรพัฒนาต่อ

### ปัญหาที่พบในโค้ด

- **`RAG_v1.py` ส่ง argument สลับกัน** — เรียก `generate_response(question, context)` แต่ฟังก์ชันรับ `(context, query_text)` ทำให้ prompt ใส่คำถามไว้ในช่อง context และใส่ context ไว้ในช่องคำถาม (แก้แล้วใน `RAG_v2.py`)
- **ตัวแปร `USER` ชนกับของระบบ** — บน macOS/Linux ค่า `USER` ใน `.env` จะถูกละเลย และโปรแกรมใช้ชื่อผู้ใช้ของเครื่องแทน ควรเปลี่ยนชื่อเป็น `DB_USER` หรือเรียก `load_dotenv(override=True)`
- **ชื่อตัวแปรฐานข้อมูลไม่ตรงกัน** — v1 ใช้ `MYDB` ส่วน v2 ใช้ `DBNAME`
- **โหลดโมเดล embedding ใหม่ทุกครั้ง** — `txt_embedding()` สร้าง `SentenceTransformer` ใหม่ทุกครั้งที่เรียก ทำให้ช้ามากตอนเพิ่มเอกสารหลายรายการ ควรโหลดครั้งเดียวไว้ระดับ module
- **`with connect_db() as conn` ของ psycopg2 ไม่ได้ปิด connection** — context manager ของ psycopg2 จัดการแค่ transaction (commit/rollback) ควรเรียก `conn.close()` เองหรือใช้ `contextlib.closing`
- **ชื่อคอลัมน์ `similarity_score` สื่อความหมายกลับกัน** — `<=>` คืนค่า *distance* (น้อย = ใกล้) ไม่ใช่ similarity ควรเปลี่ยนชื่อเป็น `distance` หรือคำนวณ `1 - (embedding <=> ...)` เป็น similarity
- **`rag_test.ipynb`** มีรหัสผ่านฐานข้อมูลเขียนไว้ในโค้ด (ควรอ่านจาก `.env`) และมี cell ท้ายไฟล์ที่ยังเขียนไม่เสร็จ (SyntaxError) ควรลบออก

### แนวทางพัฒนาต่อ

- รับคำถามและเอกสารจาก command line หรือไฟล์ แทนการกำหนดไว้ในโค้ด
- โหลดเอกสารจริง (PDF, TXT) และแบ่งเป็น chunk ก่อนทำ embedding
- สร้าง index ของ pgvector (`HNSW` หรือ `IVFFlat`) เมื่อข้อมูลมีจำนวนมาก เช่น
  `CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);`
- กำหนดเกณฑ์ distance สูงสุด เพื่อไม่ส่งเอกสารที่ไม่เกี่ยวข้องเข้า context
- ปรับ prompt ให้ LLM ตอบว่า "ไม่ทราบ" เมื่อ context ไม่มีข้อมูล และตอบเป็นภาษาเดียวกับคำถาม
- เพิ่ม `requirements.txt` และ `docker-compose.yml` สำหรับ PostgreSQL
- ทำ UI ด้วย Streamlit หรือ Gradio
