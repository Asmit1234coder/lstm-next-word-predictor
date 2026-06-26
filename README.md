# LSTM Next-Word Predictor

A next-word prediction app powered by an LSTM model, with a Streamlit interface for two modes:

- 🔮 **Next-Word Suggestions** — type a phrase and get the top predicted next words (Google-autocomplete style), with click-to-continue suggestions.
- 📝 **Sentence Generation** — give a starting phrase and let the model greedily generate a full sentence, word by word.

## Demo

| Mode | What it does |
|---|---|
| Next-Word Suggestions | Shows the top 3–8 most likely next words with their probabilities. Click any suggestion to append it and instantly see the next prediction. |
| Generate Sentence | Continuously predicts and appends the most likely next word for a chosen number of words. |

## Model Details

| Component | Detail |
|---|---|
| Architecture | Embedding → LSTM(128) → Dense (softmax) |
| Vocabulary size | 8,723 words |
| Tokenization | Word-level (Keras `Tokenizer`) |
| Max sequence length | 745 (padded with `pre` padding) |
| Training data | Custom text corpus |

## Project Structure

```
.
├── app.py              # Streamlit application
├── lstm_model.h5        # Trained LSTM model
├── tokenizer.pkl         # Fitted Keras tokenizer
├── max_len.pkl           # Max sequence length used during training
└── requirements.txt      # Python dependencies
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lstm-next-word-predictor.git
cd lstm-next-word-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

1. The input text is tokenized using the saved `tokenizer.pkl`.
2. The sequence is padded (`pre` padding) to match the model's expected input length (`max_len.pkl`).
3. The LSTM model outputs a probability distribution over the vocabulary.
4. **Suggestions mode** picks the top-k highest probability words.
5. **Generation mode** greedily picks the single highest probability word, appends it, and repeats for N words.

## Possible Improvements

- Add temperature-based sampling for more varied/creative sentence generation (instead of always picking the top word).
- Support beam search for higher-quality multi-word generation.
- Add a confidence threshold to avoid showing low-probability "noise" suggestions.

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit

## License

This project is open source and available under the [MIT License](LICENSE).
