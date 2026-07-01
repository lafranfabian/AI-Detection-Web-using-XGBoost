"""
feature_extractor.py
--------------------
Menghitung semua 15 fitur linguistik secara otomatis dari teks input.
Fitur-fitur ini HARUS konsisten dengan kolom di dataset training:

Kolom numerik (15):
    content_type, word_count, character_count, sentence_count,
    lexical_diversity, avg_sentence_length, avg_word_length,
    punctuation_ratio, flesch_reading_ease, gunning_fog_index,
    grammar_errors, passive_voice_ratio, predictability_score,
    burstiness, sentiment_score

Kolom teks (1):
    text_content
"""

import re
import math
import string
from collections import Counter
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────

# Passive voice indicators (regex sederhana berbasis pola umum)
_PASSIVE_PATTERNS = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
    re.IGNORECASE
)

# Kata-kata AI yang sangat sering muncul (approximasi predictability)
_AI_LEXICON = {
    'moreover', 'furthermore', 'additionally', 'consequently', 'therefore',
    'however', 'nevertheless', 'notwithstanding', 'subsequently', 'ultimately',
    'significantly', 'substantially', 'fundamentally', 'predominantly',
    'comprehensively', 'holistically', 'systematically', 'inherently',
    'overarching', 'paradigm', 'leverage', 'utilize', 'facilitate',
    'implement', 'robust', 'scalable', 'optimize', 'streamline', 'synergy',
    'delve', 'crucial', 'nuance', 'multifaceted', 'intricate', 'pivotal'
}

# Sentimen positif/negatif sederhana (tanpa library eksternal)
_POS_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'happy', 'joy', 'love', 'best', 'brilliant', 'superb', 'positive',
    'beautiful', 'perfect', 'outstanding', 'impressive', 'success',
    'helpful', 'effective', 'innovative', 'creative', 'inspiring'
}
_NEG_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'negative',
    'fail', 'failure', 'wrong', 'error', 'problem', 'issue', 'difficult',
    'danger', 'risk', 'threat', 'harm', 'damage', 'loss', 'crisis', 'lack'
}

# Syllable approximation: vowel clusters count
_VOWELS = set('aeiouAEIOU')


def _count_syllables(word: str) -> int:
    """Approximate syllable count per word."""
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Trailing 'e' correction
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def _is_complex_word(word: str) -> bool:
    """Gunning Fog: kata kompleks = >= 3 suku kata."""
    return _count_syllables(word) >= 3


def _tokenize_sentences(text: str):
    """Split teks menjadi kalimat berdasarkan tanda baca akhir kalimat."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_words(text: str):
    """Tokenisasi kata bersih (alfanumerik saja)."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR UTAMA
# ─────────────────────────────────────────────────────────────

def extract_features(text: str) -> Dict[str, Any]:
    """
    Hitung semua 15 fitur linguistik dari teks input.

    Returns dict dengan key yang identik dengan kolom dataset training.
    """
    text = text.strip()

    sentences = _tokenize_sentences(text)
    words = _tokenize_words(text)
    all_chars = text

    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_chars = len(all_chars)

    # ── 1. word_count ──────────────────────────────────────────
    word_count = n_words

    # ── 2. character_count ─────────────────────────────────────
    character_count = n_chars

    # ── 3. sentence_count ──────────────────────────────────────
    sentence_count = n_sentences

    # ── 4. lexical_diversity ───────────────────────────────────
    # TTR (Type-Token Ratio) × 100 untuk skala konsisten dengan dataset
    unique_words = set(words)
    lexical_diversity = (len(unique_words) / n_words) * 100

    # ── 5. avg_sentence_length ─────────────────────────────────
    sentence_lengths = [len(_tokenize_words(s)) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / n_sentences

    # ── 6. avg_word_length ─────────────────────────────────────
    avg_word_length = sum(len(w) for w in words) / n_words

    # ── 7. punctuation_ratio ───────────────────────────────────
    punct_count = sum(1 for ch in all_chars if ch in string.punctuation)
    punctuation_ratio = punct_count / max(n_chars, 1)

    # ── 8. flesch_reading_ease ─────────────────────────────────
    # Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    total_syllables = sum(_count_syllables(w) for w in words)
    flesch_reading_ease = (
        206.835
        - 1.015 * (n_words / n_sentences)
        - 84.6 * (total_syllables / n_words)
    )
    # Clamp ke range standar
    flesch_reading_ease = max(-100.0, min(121.0, flesch_reading_ease))

    # ── 9. gunning_fog_index ───────────────────────────────────
    # Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    n_complex = sum(1 for w in words if _is_complex_word(w))
    gunning_fog_index = 0.4 * (
        (n_words / n_sentences) + 100 * (n_complex / n_words)
    )

    # ── 10. grammar_errors ─────────────────────────────────────
    # Approximasi: deteksi double space, double punctuation, dll.
    grammar_errors = 0
    grammar_errors += len(re.findall(r'  +', text))                     # double space
    grammar_errors += len(re.findall(r'[.!?,;]{2,}', text))             # double punct
    grammar_errors += len(re.findall(r'\b(i)\b', text))                 # lowercase "i"
    grammar_errors += len(re.findall(r'[a-z]\. [a-z]', text))           # no capital after period

    # ── 11. passive_voice_ratio ────────────────────────────────
    passive_matches = len(_PASSIVE_PATTERNS.findall(text))
    passive_voice_ratio = passive_matches / n_sentences

    # ── 12. predictability_score ───────────────────────────────
    # Approximasi: proporsi kata AI-khas dalam teks
    lower_words = [w.lower() for w in words]
    ai_word_count = sum(1 for w in lower_words if w in _AI_LEXICON)
    predictability_score = min(1.0, ai_word_count / max(n_words, 1) * 10)

    # ── 13. burstiness ─────────────────────────────────────────
    # Variance of sentence lengths / mean (rendah = monoton = AI-khas)
    if len(sentence_lengths) > 1:
        mean_sl = sum(sentence_lengths) / len(sentence_lengths)
        variance_sl = sum((l - mean_sl) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        std_sl = math.sqrt(variance_sl)
        burstiness = std_sl / max(mean_sl, 1)
    else:
        burstiness = 0.0

    # ── 14. sentiment_score ────────────────────────────────────
    pos_count = sum(1 for w in lower_words if w in _POS_WORDS)
    neg_count = sum(1 for w in lower_words if w in _NEG_WORDS)
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words > 0:
        sentiment_score = (pos_count - neg_count) / total_sentiment_words
    else:
        sentiment_score = 0.0

    # ── 15. content_type ───────────────────────────────────────
    # Default = 0 (akan di-LabelEncode saat preprocessing)
    # Nilai 0 = kelas pertama dari LabelEncoder yang akan di-load
    # di predictor.py. Kita simpan sebagai int karena sudah di-encode.
    content_type_encoded = 0  # Placeholder, akan di-replace predictor.py

    return {
        'content_type': content_type_encoded,
        'word_count': word_count,
        'character_count': character_count,
        'sentence_count': sentence_count,
        'lexical_diversity': round(lexical_diversity, 4),
        'avg_sentence_length': round(avg_sentence_length, 4),
        'avg_word_length': round(avg_word_length, 4),
        'punctuation_ratio': round(punctuation_ratio, 6),
        'flesch_reading_ease': round(flesch_reading_ease, 4),
        'gunning_fog_index': round(gunning_fog_index, 4),
        'grammar_errors': grammar_errors,
        'passive_voice_ratio': round(passive_voice_ratio, 4),
        'predictability_score': round(predictability_score, 4),
        'burstiness': round(burstiness, 4),
        'sentiment_score': round(sentiment_score, 4),
        'text_content': text,
    }
