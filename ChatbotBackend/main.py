import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GOOGLE_API_KEY or not MONGO_URI:
    raise EnvironmentError("Missing GOOGLE_API_KEY or MONGO_URI in .env file")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["test"]
collection = db["students"]

# Allowed schema fields (not enforced here, but could be)
ALLOWED_FIELDS = {"field1", "age", "email", "status"}

# FastAPI app
app = FastAPI()

# Enable CORS (optional but useful for Postman and browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryInput(BaseModel):
    query: str

# Prompt template
PROMPT_TEMPLATE = """Convert this natural language query to a MongoDB find query.
Return ONLY valid JSON format with the query in a 'query' field.
Always include at least one filter condition using an existing field from the 'students' collection.
Assume the 'students' collection has the following fields:
- name (string)
- roll_no (number)
- email (string)
- phone_no (number)
- address (string)
- dob (date in YYYY-MM-DD format)
- class (string)
- status (string)

Example Input: Show all active users
Example Output: {{ "query": {{ "status": "active" }} }}

Input: {user_input}
Output:"""

# Utility: Clean LLM response
def clean_json_response(response: str) -> dict:
    try:
        cleaned = response.strip().replace('```json', '').replace('```', '')
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

# Route: POST /query
@app.post("/query")
async def handle_query(input: QueryInput):
    try:
        prompt = PROMPT_TEMPLATE.format(user_input=input.query)
        response = model.generate_content(prompt)
        result = clean_json_response(response.text)
        mongo_query = result.get("query", {})

        if not mongo_query:
            raise HTTPException(status_code=400, detail="Empty or invalid query generated.")

        cursor = collection.find(mongo_query).limit(5)
        results = list(cursor)

        if not results:
            return JSONResponse(content={"message": "No documents found."}, status_code=200)

        json_docs = json.loads(json_util.dumps(results))
        return {"results": json_docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
