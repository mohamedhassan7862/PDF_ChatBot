from fastapi import FastAPI, UploadFile, File, Form
from pdf_chatbot import PDFChatbot
import tempfile

app = FastAPI()
chatbots = {}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # Create chatbot and store it using filename as key
    chatbot = PDFChatbot(pdf_path)
    chatbots[file.filename] = chatbot
    return {"message": f"Chatbot initialized for {file.filename}"}

@app.post("/ask/")
async def ask_question(filename: str = Form(...), question: str = Form(...)):
    if filename not in chatbots:
        return {"error": "Chatbot not found. Upload the PDF first."}
    response = chatbots[filename].ask(question)
    return {"answer": response}
