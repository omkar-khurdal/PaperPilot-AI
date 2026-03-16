import re

chat_history = []
conversation_topics = []


def add_user_message(message):
    chat_history.append({
        "role": "user",
        "content": message
    })


def add_ai_message(message):
    chat_history.append({
        "role": "assistant",
        "content": message
    })


def get_history():
    return chat_history


def get_recent_history(limit=4):
    return chat_history[-limit:]


def clear_history():
    chat_history.clear()
    conversation_topics.clear()


def set_conversation_topics(topics):
    """
    Store current active conversation topics.
    Example:
    ["machine learning", "deep learning"]
    """
    global conversation_topics
    conversation_topics = topics[:]


def get_conversation_topics():
    return conversation_topics[:]