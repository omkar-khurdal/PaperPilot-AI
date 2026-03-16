import re


def clean_text(text):
    """
    Clean raw text extracted from PDF
    """

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove newlines
    text = text.replace("\n", " ")

    # Remove weird characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()