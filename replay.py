import ast
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)


def validate_response(text):
    blocks = list(re.finditer(
        r"^[ \t]*```python[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$",
        text,
        re.MULTILINE | re.DOTALL,
    ))
    if not blocks:
        raise ValueError("AI response is missing a fenced Python code block.")

    prose = re.sub(
        r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*\r?$",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not any(
        line.strip(" \t#*:").lower() not in {"", "explanation", "code"}
        for line in prose.splitlines()
    ):
        raise ValueError("AI response is missing a plain-English explanation.")

    for block in blocks:
        code = block.group(1)
        if not code.strip():
            raise ValueError("AI response contains an empty Python code block.")
        try:
            ast.parse(code)
        except SyntaxError as error:
            raise ValueError(
                f"AI response contains invalid Python (line {error.lineno}: {error.msg})."
            ) from error


def request_strategy(client, idea):
    return client.responses.create(
        model="gpt-5-mini",
        instructions=(
            "Turn the user's trading idea into Python strategy code. "
            "Return two sections: Explanation (plain English, including "
            "assumptions) and Code (inside a fenced ```python code block). "
            "State signal timing, fill timing and prices, and return accounting; "
            "use the same conventions in the explanation, trade log, and code. "
            "For next-bar execution, leave a signal unfilled when there is no "
            "following bar; never fall back to that signal bar's Close. "
            "Leave unfinished positions open unless the user requests a final "
            "sale; record missing exits and realized trade returns as None, "
            "and distinguish unrealized P&L from realized results. "
            "For any supported positive quantity, calculate trade P&L as "
            "(exit_or_mark_price - entry_price) * qty and the fractional "
            "return as P&L / (entry_price * qty). Quantity changes currency "
            "P&L, not the fractional return for the same price move. Apply "
            "this to both realized and unrealized returns, and validate "
            "denominators. State the capital denominator for any portfolio "
            "summary return. Calculate win/loss summaries using boolean "
            "masks on realized P&L, with defined handling of empty groups. "
            "Wait for complete rolling windows before comparing indicators, "
            "unless the user explicitly requests a different warm-up rule. "
            "Validate rolling windows as positive integers before calculation; "
            "for fast and slow moving averages require slow >= fast. "
            "Provide one complete implementation without placeholder trade "
            "logic or discarded backtest loops. Check all execution paths, "
            "including final-bar signals and open positions, for contradictions "
            "with the explanation and the user's requirements before responding. "
            "List any required dependencies. "
            "Do not execute the code or place trades."
        ),
        input=idea,
    )


def main():
    idea = input("Describe a trading idea: ").strip()
    if not idea:
        print("Please enter a trading idea.")
        return
    print(f"Your trading idea: {idea}")

    try:
        load_dotenv(Path(__file__).with_name(".env"))
    except (OSError, UnicodeError):
        print("Error: Could not read .env. Check its permissions and encoding.")
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAI_API_KEY is missing. Add it to your local .env file.")
        return

    try:
        with OpenAI(api_key=api_key, max_retries=0, timeout=120.0) as client:
            response = request_strategy(client, idea)
        if response.status != "completed" or not response.output_text.strip():
            print("Error: OpenAI returned an incomplete or empty response.")
            return
        try:
            validate_response(response.output_text)
        except ValueError as error:
            print(f"Error: {error}")
        else:
            print(response.output_text)
    except AuthenticationError:
        print("Error: OpenAI rejected the API key. Check OPENAI_API_KEY in .env.")
    except RateLimitError:
        print("Error: OpenAI rate limit or quota reached. Check billing or try later.")
    except APITimeoutError:
        print("Error: OpenAI request timed out. Try again later.")
    except APIConnectionError:
        print("Error: Could not reach OpenAI. Check your connection or try later.")
    except OpenAIError:
        print("Error: OpenAI request failed. Check model access or try later.")


if __name__ == "__main__":
    main()
