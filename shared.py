from typing import Iterable, List, Literal, Union
import torch
from rich import box
from rich.align import Align
from rich.console import Console
from rich.table import Table
from torch import tensor
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-community/openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-community/openai-gpt')

demo_embedding_table = tensor(
    [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    ]
)


def show_token_mapping(
    direction: Literal['token->id', 'id->token'],
    tokenizer,
    data: Union[str, Iterable[int]],
) -> None:
    """
    Display token mappings in a formatted table.

    Args:
        direction: Either 'token->id' or 'id->token'
        tokenizer: The tokenizer to use for conversions
        data: Text string (for 'token->id') or token IDs (for 'id->token')
    """
    console = Console()
    table = Table(show_header=True, header_style='', box=box.HEAVY)

    if direction == 'token->id':
        table.add_column(Align.center('Token'), justify='left')
        table.add_column(Align.center('Token ID'), justify='right')

        # Tokenize the input text
        result = tokenizer(data, return_attention_mask=False)
        token_ids = result['input_ids']

        # Decode each token individually to get the token strings
        for token_id in token_ids:
            token_str = tokenizer.decode([token_id])
            table.add_row(f"'{token_str}'", str(token_id))

    elif direction == 'id->token':
        table.add_column(Align.center('Token ID'), justify='left')
        table.add_column(Align.center('Token'), justify='left')

        results = [(token_id, tokenizer.decode([token_id])) for token_id in data]
        for token_id, token_str in results:
            table.add_row(str(token_id), f"'{token_str}'")

    console.print(table)


def show_probabilities(logits: torch.Tensor, probabilities: torch.Tensor, vocab: List[str]) -> None:
    """
    Display logits and their corresponding probabilities in a formatted table.

    Args:
        logits: Raw model outputs (logits)
        probabilities: Computed probability distribution from softmax
        vocab: List of vocabulary tokens corresponding to each logit
    """
    console = Console()

    table = Table(show_header=True, header_style='', box=box.HEAVY)
    table.add_column(Align.center('Token'), justify='left')
    table.add_column(Align.center('Logit'), justify='right')
    table.add_column(Align.center('Probability'), justify='right')

    for word, logit, prob in zip(vocab, logits, probabilities):
        table.add_row(
            f"'{word}'",
            f'{logit:.2f}',
            f'{prob:.4f}',
        )

    console.print(table)
