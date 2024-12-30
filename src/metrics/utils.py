import Levenshtein


def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Character Error Rate (CER) between target and predicted text.
    CER = (substitutions + insertions + deletions) / length of target text

    Args:
        target_text (str): Ground truth text
        predicted_text (str): ASR model prediction, can be None or empty
    Returns:
        float: Character Error Rate
    """
    # LOG
    print(f"target_text - {target_text}")
    print(f"predicted_text - {predicted_text}")
    # Handle None case in predicted text
    if predicted_text is None:
        predicted_text = ""

    # Convert inputs to strings and handle empty cases
    target_text = str(target_text)
    predicted_text = str(predicted_text)

    # If target is empty
    if not target_text:
        return 1.0 if predicted_text else 0.0

    # Calculate Levenshtein distance
    distance = Levenshtein.distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Word Error Rate (WER) between target and predicted text.
    WER = (substitutions + insertions + deletions) / number of words in target

    Args:
        target_text (str): Ground truth text
        predicted_text (str): ASR model prediction, can be None or empty
    Returns:
        float: Word Error Rate
    """
    # Handle None case in predicted text
    if predicted_text is None:
        predicted_text = ""

    # Convert inputs to strings
    target_text = str(target_text)
    predicted_text = str(predicted_text)

    # Split into words and handle empty cases
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    # If target is empty
    if not target_words:
        return 1.0 if predicted_words else 0.0

    # Calculate Levenshtein distance at word level
    distance = Levenshtein.distance(target_words, predicted_words)
    return distance / len(target_words)
