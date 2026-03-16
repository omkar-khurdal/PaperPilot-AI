import re

from app.memory import (get_conversation_topics, get_history,
                        set_conversation_topics)

PRONOUNS = {
    "it", "this", "that", "they", "them", "its", "their",
    "these", "those", "former", "latter"
}

COMPARISON_WORDS = {
    "compare", "difference", "differences", "better", "best", "worse",
    "which one", "which is better", "versus", "vs"
}

TOPIC_PATTERNS = [
    r"^what is\s+(.+)$",
    r"^what are\s+(.+)$",
    r"^who is\s+(.+)$",
    r"^who are\s+(.+)$",
    r"^define\s+(.+)$",
    r"^explain\s+(.+)$",
    r"^types of\s+(.+)$",
    r"^applications of\s+(.+)$",
    r"^advantages of\s+(.+)$",
    r"^disadvantages of\s+(.+)$",
    r"^tell me about\s+(.+)$",
    r"^give me\s+(.+)$",
]

COMPARE_PATTERNS = [
    r"^compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
    r"^difference between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
    r"^what is the difference between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
    r"^what's the difference between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
    r"^tell me the difference between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
    r"^(.+?)\s+(?:vs|versus)\s+(.+)$",
]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_pronoun(question: str) -> bool:
    words = re.findall(r"\b[a-zA-Z]+\b", normalize_text(question))
    return any(word in PRONOUNS for word in words)

def is_short_followup(question: str) -> bool:
    q = normalize_text(question).strip("?.!,;:")
    short_patterns = {
        "why", "how", "how so", "explain", "explain more",
        "advantages", "disadvantages", "applications",
        "uses", "limitations", "example", "examples"
    }
    return q in short_patterns

def extract_single_topic(question: str):
    q = normalize_text(question).rstrip("?.!,;:")

    for pattern in TOPIC_PATTERNS:
        match = re.match(pattern, q)
        if match:
            topic = match.group(1).strip(" ?.!,;:")
            if topic and topic not in PRONOUNS:
                return topic

    return None


def extract_comparison_topics(question: str):
    q = normalize_text(question).rstrip("?.!,;:")

    for pattern in COMPARE_PATTERNS:
        match = re.match(pattern, q)
        if match:
            t1 = match.group(1).strip(" ?.!,;:")
            t2 = match.group(2).strip(" ?.!,;:")

            # remove leading filler words if they leak in
            t1 = re.sub(r"^(the\s+)?", "", t1).strip()
            t2 = re.sub(r"^(the\s+)?", "", t2).strip()

            if t1 and t2:
                return [t1, t2]

    return None

def is_comparison_followup(question: str) -> bool:
    q = normalize_text(question)

    patterns = [
        "which one",
        "which of them",
        "which is better",
        "which one is better",
        "which is best",
        "which works better",
        "which performs better",
        "which is faster",
        "which is more suitable",
        "these two",
        "among them",
        "between them",
        "former",
        "latter",
    ]

    return any(p in q for p in patterns)

def update_topics_from_question(question: str):
    """
    Detect topic(s) from explicit user question and store them.
    """
    compare_topics = extract_comparison_topics(question)
    if compare_topics:
        set_conversation_topics(compare_topics)
        return compare_topics

    single_topic = extract_single_topic(question)
    if single_topic:
        set_conversation_topics([single_topic])
        return [single_topic]

    return get_conversation_topics()


def get_last_explicit_topic():
    history = get_history()

    for msg in reversed(history):
        if msg.get("role") != "user":
            continue

        comparison = extract_comparison_topics(msg.get("content", ""))
        if comparison:
            return comparison

        single = extract_single_topic(msg.get("content", ""))
        if single:
            return [single]

    return []


def rewrite_query(current_question: str) -> str:
    """
    Improved query rewriting with:
    - explicit topic tracking
    - comparison follow-ups
    - former/latter
    - short vague follow-ups like:
      explain more, why, how, advantages, disadvantages, examples
    """
    current = current_question.strip()
    if not current:
        return current

    current_lower = normalize_text(current)

    explicit_single = extract_single_topic(current)
    explicit_compare = extract_comparison_topics(current)

    # store explicit topics immediately
    if explicit_compare:
        set_conversation_topics(explicit_compare)
        return current

    if explicit_single:
        set_conversation_topics([explicit_single])
        return current

    topics = get_conversation_topics()
    if not topics:
        topics = get_last_explicit_topic()

    if not topics:
        return current

    # CASE 1: former / latter
    if len(topics) >= 2:
        t1, t2 = topics[0], topics[1]
        rewritten = current

        if re.search(r"\bformer\b", current_lower):
            rewritten = re.sub(r"\bformer\b", t1, rewritten, flags=re.IGNORECASE)

        if re.search(r"\blatter\b", current_lower):
            rewritten = re.sub(r"\blatter\b", t2, rewritten, flags=re.IGNORECASE)

        if rewritten != current:
            return rewritten

    # CASE 2: comparison follow-up
    comparison_patterns = [
        "which one",
        "which of them",
        "which works better",
        "which is better",
        "which is best",
        "which is used",
        "among them",
        "between them",
        "these two",
    ]

    if len(topics) >= 2 and any(p in current_lower for p in comparison_patterns):
        t1, t2 = topics[0], topics[1]
        clean_question = current.strip().rstrip("?.!,;:")
        clean_question = re.sub(r"\bwhich one\b", "which", clean_question, flags=re.IGNORECASE)
        clean_question = re.sub(r"\bwhich of them\b", "which", clean_question, flags=re.IGNORECASE)
        return f"{clean_question}: {t1} or {t2}?"

    # CASE 3: short vague follow-up
    short_followups = {
        "explain more",
        "explain",
        "why",
        "how",
        "advantages",
        "what are advantages",
        "what are the advantages",
        "disadvantages",
        "what are disadvantages",
        "what are the disadvantages",
        "applications",
        "uses",
        "limitations",
        "examples",
        "give examples",
    }

    if current_lower in short_followups:
        if len(topics) >= 2:
            return f"{current.strip().rstrip('?.!,;:')} of {topics[0]} and {topics[1]}"
        return f"{current.strip().rstrip('?.!,;:')} of {topics[0]}"

    # CASE 4: pronoun follow-up
    if contains_pronoun(current):
        replacement = " and ".join(topics) if len(topics) >= 2 else topics[0]

        rewritten = current
        rewritten = re.sub(r"\bit\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthis\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthat\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthey\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthem\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bits\b", f"{replacement}'s", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\btheir\b", f"{replacement}'s", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthese\b", replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthose\b", replacement, rewritten, flags=re.IGNORECASE)

        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten

    return current