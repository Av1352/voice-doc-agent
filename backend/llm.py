import re
from typing import Generator
from anthropic import Anthropic

def stream_response(query: str, context_chunks: list[dict]) -> Generator[str, None, None]:
    """
    Combines retrieved context chunks with a user query, invokes Anthropic's Claude 
    using the specified system instructions, and streams the response back token by 
    token. Assembles the tokens into complete sentences using regex boundary detection 
    and yields them one at a time as soon as they are fully formed.
    """
    # Anthropic client will automatically look for the ANTHROPIC_API_KEY environment variable.
    client = Anthropic()

    # Build the user prompt combining context and the query
    context_text = "\n\n".join([f"- {c.get('content', '')}" for c in context_chunks])
    
    user_prompt = (
        f"Use the following piece of context to answer the user's query.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Query: {query}"
    )

    # Instruct Claude to answer in plain, simple, short sentences 
    system_prompt = (
        "Answer the user's query based ONLY on the provided context. "
        "You must answer in plain, simple language that is perfectly easy to understand. "
        "Strict Rule: Every single sentence you write MUST be short and under 20 words each. "
        "Do not write any long, run-on sentences."
    )

    # We use a non-greedy regex to find sentence-ending punctuation followed by whitespace.
    # Searching for at least one whitespace ensures we don't accidentally split abbreviations 
    # if they sit right at the edge of the incoming stream sequence without a space yet.
    sentence_boundary_regex = re.compile(r'[.!?]+\s+')
    buffer = ""

    # Invoke the Anthropic stream API 
    with client.messages.stream(
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
        model="claude-sonnet-4-20250514",
    ) as stream:
        for text_token in stream.text_stream:
            buffer += text_token
            
            # Continuously scan the current buffer for sentence boundaries
            while True:
                match = sentence_boundary_regex.search(buffer)
                if match:
                    end_idx = match.end()
                    # Extract the full sentence and strip any outlying edge whitespace
                    sentence = buffer[:end_idx].strip()
                    
                    if sentence:
                        sentence = re.sub(r'[*#]', '', sentence).strip()
                        yield sentence
                        
                    # Advance the buffer past the detected boundary chunk
                    buffer = buffer[end_idx:]
                else:
                    break

    # Once the stream completely wraps up, flush anything remaining in the buffer.
    remaining = buffer.strip()
    if remaining:
        remaining = re.sub(r'[*#]', '', remaining).strip()
        yield remaining

if __name__ == "__main__":
    # Test stub
    pass
