import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered",
)

# ----------------------------------------------------------------------------
# MINIMAL CUSTOM STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { padding-top: 1.5rem; }

        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .app-subtitle {
            color: #888;
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }

        /* Google-style suggestion dropdown */
        .suggest-box {
            background: #25262b;
            border: 1px solid #3a3b40;
            border-radius: 14px;
            overflow: hidden;
            margin-top: 0.4rem;
        }

        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            color: #e8eaed !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            padding: 0.85rem 1rem !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
            background: #33343a !important;
            color: #ffffff !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p {
            font-size: 1rem !important;
        }

        .suggest-row-wrap {
            border-bottom: 1px solid #34353a;
        }
        .suggest-row-wrap:last-child {
            border-bottom: none;
        }

        .suggest-prob {
            color: #8a8d93;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            height: 100%;
            justify-content: flex-end;
            padding-right: 0.3rem;
        }

        .generated-box {
            background: #25262b;
            border: 1px solid #3a3b40;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            font-size: 1.1rem;
            line-height: 1.7;
        }
        .seed-part { color: #e8eaed; }
        .gen-part { color: #4dabf7; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LOAD MODEL + ARTIFACTS (cached so this only runs once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word


model, tokenizer, max_len, index_word = load_artifacts()
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index 0


# ----------------------------------------------------------------------------
# CORE PREDICTION LOGIC
# ----------------------------------------------------------------------------
def get_next_word_probs(seed_text: str) -> np.ndarray:
    """Returns the softmax probability vector for the next word."""
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    preds = model.predict(seq, verbose=0)[0]
    return preds


def top_k_words(seed_text: str, k: int = 5):
    """Returns list of (word, probability) for the top-k most likely next words."""
    preds = get_next_word_probs(seed_text)
    top_indices = preds.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        if idx == 0:
            continue  # skip padding index
        word = index_word.get(idx)
        if word:
            results.append((word, float(preds[idx])))
    return results


def generate_sentence(seed_text: str, n_words: int) -> str:
    """Greedily appends the most likely next word, n_words times."""
    text = seed_text
    for _ in range(n_words):
        preds = get_next_word_probs(text)
        next_idx = int(np.argmax(preds))
        next_word = index_word.get(next_idx, "")
        if not next_word:
            break
        text += " " + next_word
    return text


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<div class="app-title">✍️ Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="app-subtitle">LSTM model · {vocab_size:,} word vocabulary · sequence length {max_len}</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODE SELECTOR
# ----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Next-Word Suggestions", "📝 Generate Sentence"])

# ----------------------------------------------------------------------------
# TAB 1: TOP-K NEXT WORD SUGGESTIONS
# ----------------------------------------------------------------------------
with tab1:
    st.write("Type a phrase and see the model's top predicted next words — click a suggestion to keep building the sentence, just like Google search.")

    if "seed_1" not in st.session_state:
        st.session_state.seed_1 = ""
    if "show_suggestions" not in st.session_state:
        st.session_state.show_suggestions = False

    st.text_input(
        "Your text",
        placeholder="e.g. once upon a",
        key="seed_1",
    )

    k = st.slider("Number of suggestions", min_value=3, max_value=8, value=4)

    if st.button("Predict next word", type="primary", use_container_width=True):
        st.session_state.show_suggestions = True

    if st.session_state.show_suggestions:
        current_text = st.session_state.seed_1

        if not current_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Predicting..."):
                suggestions = top_k_words(current_text, k=k)

            if not suggestions:
                st.info("No confident predictions found — try a longer phrase.")
            else:
                st.markdown("#### Suggestions")
                st.markdown('<div class="suggest-box">', unsafe_allow_html=True)
                for i, (word, prob) in enumerate(suggestions):
                    st.markdown('<div class="suggest-row-wrap">', unsafe_allow_html=True)
                    col_btn, col_prob = st.columns([6, 1])
                    with col_btn:
                        if st.button(f"🔍  {word}", key=f"sugg_{i}_{word}", use_container_width=True):
                            st.session_state.seed_1 = current_text.rstrip() + " " + word
                            st.rerun()
                    with col_prob:
                        st.markdown(
                            f'<div class="suggest-prob">{prob*100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.caption("💡 Click any suggestion to add it and see the next prediction.")

# ----------------------------------------------------------------------------
# TAB 2: SENTENCE GENERATION
# ----------------------------------------------------------------------------
with tab2:
    st.write("Give the model a starting phrase and let it keep writing.")

    seed_text_2 = st.text_input(
        "Starting phrase",
        placeholder="e.g. the future of technology is",
        key="seed_2",
    )

    n_words = st.slider("Words to generate", min_value=1, max_value=30, value=10)

    if st.button("Generate sentence", type="primary", use_container_width=True):
        if not seed_text_2.strip():
            st.warning("Please enter a starting phrase first.")
        else:
            with st.spinner("Generating..."):
                full_text = generate_sentence(seed_text_2, n_words)

            generated_part = full_text[len(seed_text_2):]
            st.markdown("#### Result")
            st.markdown(
                f"""
                <div class="generated-box">
                    <span class="seed-part">{seed_text_2}</span><span class="gen-part">{generated_part}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("Built with an LSTM (Embedding → LSTM(128) → Dense softmax) trained on a custom corpus.")