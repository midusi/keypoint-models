from typing import Callable


def edit_distance(prediction_tokens: list[str], reference_tokens: list[str], ignore_tokens: list[str] = []) -> int:
    """Standard dynamic programming algorithm to compute the edit distance adding a list of tokens to ignore
    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
        ignore_tokens: Tokens to ignore substitutions and insertions
    Returns:
        Edit distance between the predicted sentence and the reference sentence, ignoring insertions and substitutions of ignore_tokens
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(1, len(reference_tokens) + 1):
        dp[0][j] = dp[0][j - 1] + (1 if reference_tokens[j - 1] not in ignore_tokens else 0)
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + (1 if reference_tokens[j - 1] not in ignore_tokens else 0),
                dp[i - 1][j - 1] + (1 if prediction_tokens[i - 1] != reference_tokens[j - 1]
                                        and reference_tokens[j - 1] not in ignore_tokens else 0)
            )
    return dp[-1][-1]

def wer_n(prediction: str, target: str, ignore_tokens: list[str], tokenizer: Callable[[str], list[str]] = str.split) -> float:
    return edit_distance(tokenizer(prediction), tokenizer(target), ignore_tokens) / len(tokenizer(target))