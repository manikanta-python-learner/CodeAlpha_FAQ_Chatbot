# CodeAlpha – FAQ Chatbot using NLP (Task 2)

A desktop FAQ chatbot built with Python and Tkinter. It preprocesses text using
NLTK, converts FAQ questions into TF-IDF vectors, and uses cosine similarity
to retrieve the closest matching answer.

The included FAQ dataset is based on the CodeAlpha AI Internship task instructions.

## Features
- Tkinter chat interface
- 22 FAQ entries with alternative phrasings
- NLTK preprocessing: lowercase, tokenization, stop-word removal, Porter stemming
- TF-IDF vectorization with unigram and bigram features
- Cosine similarity FAQ retrieval
- Similarity threshold for unrelated questions
- Suggested questions and clear chat
- Match label and similarity score

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## How it works
1. Load FAQ entries from `faqs.json`.
2. NLTK normalizes, tokenizes, removes stop words, and stems words.
3. Scikit-learn creates TF-IDF vectors for the FAQ questions.
4. The user's question is processed with the same pipeline.
5. Cosine similarity finds the closest FAQ.
6. The best answer is displayed when the score reaches the threshold; otherwise, the bot asks the user to rephrase.

## Files
- `main.py` – application and NLP matching
- `faqs.json` – FAQ questions, variations, and answers
- `requirements.txt` – dependencies
- `README.md` – setup and usage
- `.gitignore` – ignored files

## Customizing FAQs
Edit `faqs.json`. Each entry has `question`, `variations`, and `answer`.
Keep the JSON syntax valid.

## Limitations
This is a retrieval-based FAQ chatbot, not a generative AI chatbot. It answers
only from the supplied FAQ dataset and does not independently verify policy changes.
The similarity threshold may need tuning for a different dataset.

## GitHub repository name
`CodeAlpha_FAQ_Chatbot`

After creating an empty GitHub repository:
```bash
git init
git add .
git commit -m "Complete CodeAlpha FAQ Chatbot Task 2"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/CodeAlpha_FAQ_Chatbot.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## CodeAlpha Task 2 mapping
| Instruction | Implementation |
|---|---|
| Collect FAQs | `faqs.json` |
| NLP preprocessing | NLTK tokenizer, stop-word removal, Porter stemming |
| Similarity matching | TF-IDF + cosine similarity |
| Display best answer | Tkinter chat UI |
| Optional chat UI | Implemented |

CodeAlpha AI Internship – Task 2
