"""
ask command for KuroCode.
"""

import sys
import json
import asyncio
import click

from kurocode.types import CliContext
from kurocode.core.session import Session
from kurocode.core.renderer import OutputFormat
from kurocode.infra.openrouter_client import OpenRouterClient

@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--model",
    default="openai/gpt-4o-mini",
    help="Model to use for the response.",
)
@click.pass_obj
def ask_cmd(ctx: CliContext, prompt: str | None, model: str) -> None:
    """Ask a single question and get a response."""
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            ctx.renderer.error("No prompt provided. Please pass a prompt or pipe to stdin.")
            sys.exit(1)
            
    if not prompt:
        ctx.renderer.error("Empty prompt provided.")
        sys.exit(1)

    asyncio.run(run_ask(ctx, prompt, model))


from kurocode.exceptions import RateLimitError
from kurocode.core.model_registry import ModelRegistry

async def run_ask(ctx: CliContext, prompt: str, model: str) -> None:
    session = Session(model_id=model)
    session.add_user_message(prompt)
    messages = session.to_openrouter_messages()
    
    tried_models = set()
    current_model = model

    async with OpenRouterClient(ctx.config) as client:
        while True:
            try:
                if ctx.no_stream:
                    resp = await client.chat(messages=messages, model=current_model)
                    content = resp.choices[0].message.content
                    
                    # Pipe-friendly JSON output for `jq .content`
                    if ctx.renderer._fmt == OutputFormat.JSON:
                        ctx.renderer._console.print(
                            json.dumps({"content": content}),
                            markup=False,
                            highlight=False
                        )
                    else:
                        ctx.renderer.stream_token(content)
                        ctx.renderer.end_stream()
                else:
                    try:
                        async for chunk in client.stream_chat(messages=messages, model=current_model):
                            delta = chunk.choices[0].delta.content
                            if delta:
                                ctx.renderer.stream_token(delta)
                        ctx.renderer.end_stream()
                    except KeyboardInterrupt:
                        ctx.renderer.end_stream()
                        ctx.renderer.error("\n[Interrupted]")
                break  # Success
            except RateLimitError:
                tried_models.add(current_model)
                registry = ModelRegistry()
                fallback = await registry.get_fallback_model(current_model, tried_models)
                if not fallback:
                    ctx.renderer.error(f"\n[Model {current_model} is exhausted and no fallback models are available.]")
                    sys.exit(1)
                
                ctx.renderer.info(f"\n[Model {current_model} is exhausted. Falling back to {fallback}...]")
                current_model = fallback
