import openai
import asyncio
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from typing import Any

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    

    openai.api_key = ""
    completion = openai.Completion()
    
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

predictions = asyncio.run(
    dispatch_openai_requests(
        messages_list = [
            [{"role": "user", "content": "Please extract the following information from the email below"}],
            [{"role": "user", "content": "Subject (key: subject) Sender's name (key: sender_name) Sender's email address (key: sender_email) Recipient's name (key: recipient_name) Recipient's email address (key: recipient_email) Date sent (key: date_sent) Body text (key: body_text) A short summary of the email (key: summary)"}],    
            [{"role": "user", "content": "Make sure fields 1 to 6 are answered very short, e.g. for the sender's name just say the name. Please answer in JSON machine-readable format, using the keys from above. Format the ouput as JSON object called 'results'. Pretty print the JSON and make sure that is properly closed at the end."}], 
        ],
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=200,
        top_p=1.0,
    )
)

for i, x in enumerate(predictions):
    print(f"Response {i}: {x['choices'][0]['message']['content']}\n\n")
