"""
CodeAlpha Task 2: FAQ Chatbot using NLP
Tkinter GUI | NLTK preprocessing | TF-IDF | Cosine Similarity
"""
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

try:
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    PorterStemmer = TfidfVectorizer = cosine_similarity = None

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "faqs.json"
SIMILARITY_THRESHOLD = 0.22


class FAQEngine:
    def __init__(self, faq_data):
        if not all((PorterStemmer, TfidfVectorizer, cosine_similarity)):
            raise ImportError("Install dependencies with: python -m pip install -r requirements.txt")
        self.faq_data = faq_data
        self.stemmer = PorterStemmer()
        self.stop_words = {
            "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
            "being", "do", "does", "did", "to", "of", "in", "on", "for", "and",
            "or", "with", "me", "my", "i", "you", "your", "we", "our", "it",
            "this", "that", "what", "how", "can", "could", "should", "would",
            "please", "tell", "about"
        }
        self.questions, self.answers, self.labels = [], [], []
        for item in faq_data:
            for question in [item["question"]] + item.get("variations", []):
                self.questions.append(question)
                self.answers.append(item["answer"])
                self.labels.append(item["question"])
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess, token_pattern=None,
            lowercase=False, ngram_range=(1, 2)
        )
        self.question_matrix = self.vectorizer.fit_transform(self.questions)

    def preprocess(self, text):
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [
            self.stemmer.stem(token) for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

    def get_answer(self, question):
        if not question.strip():
            return None, 0.0, None
        vector = self.vectorizer.transform([question])
        scores = cosine_similarity(vector, self.question_matrix).flatten()
        index = int(scores.argmax())
        score = float(scores[index])
        if score < SIMILARITY_THRESHOLD:
            return (
                "Sorry, I couldn't find a matching FAQ. Please rephrase your "
                "question or ask about CodeAlpha tasks, GitHub, LinkedIn, or submissions.",
                score, None
            )
        return self.answers[index], score, self.labels[index]


class FAQChatbotApp:
    def __init__(self, root):
        self.root = root
        root.title("CodeAlpha | FAQ Chatbot using NLP")
        root.geometry("900x680")
        root.minsize(720, 560)
        root.configure(bg="#f3f6fb")
        try:
            with open(FAQ_FILE, encoding="utf-8") as f:
                self.engine = FAQEngine(json.load(f))
        except Exception as exc:
            messagebox.showerror("Startup error", f"Could not load chatbot data:\n{exc}")
            raise
        self.build_ui()
        self.add_message("bot", "Hello! I'm the CodeAlpha Internship FAQ Assistant. "
                          "Ask me about tasks, project requirements, GitHub, LinkedIn, certificates, or submissions.")

    def build_ui(self):
        header = tk.Frame(self.root, bg="#172554", padx=22, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="CODEALPHA FAQ CHATBOT", font=("Segoe UI", 20, "bold"),
                 fg="white", bg="#172554").pack(anchor="w")
        tk.Label(header, text="NLP • NLTK • TF-IDF • Cosine Similarity",
                 font=("Segoe UI", 10), fg="#bfdbfe", bg="#172554").pack(anchor="w", pady=(5, 0))

        body = tk.Frame(self.root, bg="#f3f6fb", padx=18, pady=16)
        body.pack(fill="both", expand=True)
        chat = tk.Frame(body, bg="white", bd=1, relief="solid")
        chat.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(chat, bg="white", highlightthickness=0)
        scroll = ttk.Scrollbar(chat, orient="vertical", command=self.canvas.yview)
        self.messages = tk.Frame(self.canvas, bg="white")
        self.messages.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window = self.canvas.create_window((0, 0), window=self.messages, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scroll.pack(side="right", fill="y")

        suggestions = tk.Frame(body, bg="#f3f6fb")
        suggestions.pack(fill="x", pady=(10, 6))
        tk.Label(suggestions, text="Try asking:", bg="#f3f6fb", fg="#475569",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        for q in ("How many tasks should I complete?", "How do I submit my project?", "What is Task 2?"):
            ttk.Button(suggestions, text=q, command=lambda question=q: self.ask_suggestion(question)).pack(side="left", padx=3)

        row = tk.Frame(body, bg="#f3f6fb")
        row.pack(fill="x", pady=(5, 0))
        self.entry = tk.Entry(row, font=("Segoe UI", 12), relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True, ipady=10)
        self.entry.bind("<Return>", lambda e: self.send_message())
        self.entry.focus_set()
        ttk.Button(row, text="Send", command=self.send_message).pack(side="left", padx=(8, 0), ipady=5)

        footer = tk.Frame(self.root, bg="#e2e8f0", padx=18, pady=8)
        footer.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value=f"Loaded {len(self.engine.faq_data)} FAQs")
        tk.Label(footer, textvariable=self.status, bg="#e2e8f0", fg="#334155").pack(side="left")
        ttk.Button(footer, text="Clear chat", command=self.clear_chat).pack(side="right")

    def add_message(self, sender, message):
        user = sender == "user"
        row = tk.Frame(self.messages, bg="white", padx=10, pady=7)
        row.pack(fill="x", anchor="e" if user else "w")
        color = "#dbeafe" if user else "#f1f5f9"
        bubble = tk.Frame(row, bg=color, padx=12, pady=9)
        bubble.pack(anchor="e" if user else "w")
        tk.Label(bubble, text="You" if user else "FAQ Assistant", bg=color,
                 fg="#1e40af" if user else "#166534", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(bubble, text=message, bg=color, fg="#0f172a", font=("Segoe UI", 10),
                 justify="left", wraplength=650, anchor="w").pack(anchor="w", pady=(3, 0))
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def send_message(self):
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, "end")
        self.add_message("user", question)
        answer, score, matched = self.engine.get_answer(question)
        self.add_message("bot", answer)
        self.status.set(f"Matched: {matched} | Similarity: {score:.2f}" if matched
                        else f"No confident match | Similarity: {score:.2f}")

    def ask_suggestion(self, question):
        self.entry.delete(0, "end")
        self.entry.insert(0, question)
        self.send_message()

    def clear_chat(self):
        for widget in self.messages.winfo_children():
            widget.destroy()
        self.add_message("bot", "Chat cleared. Ask me about CodeAlpha internship tasks, requirements, GitHub, LinkedIn, or submission.")
        self.status.set(f"Loaded {len(self.engine.faq_data)} FAQs")


if __name__ == "__main__":
    root = tk.Tk()
    app = FAQChatbotApp(root)
    root.mainloop()
