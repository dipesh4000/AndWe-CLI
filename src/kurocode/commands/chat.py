"""
chat command for KuroCode.
"""

import sys
import asyncio
from pathlib import Path
import click

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from kurocode.types import CliContext
from kurocode.core.session import Session
from kurocode.infra.openrouter_client import OpenRouterClient
from kurocode.infra.store import ConversationStore


@click.command()
@click.option(
    "--model",
    default="openrouter/owl-alpha",
    help="Model to use for the response.",
)
@click.option(
    "--resume",
    help="Resume an existing conversation by ID.",
)
@click.pass_obj
def chat_cmd(ctx: CliContext, model: str, resume: str | None) -> None:
    """Start an interactive chat session."""
    asyncio.run(run_chat(ctx, model, resume))


async def run_chat(ctx: CliContext, model: str, resume: str | None) -> None:
    history_file = Path.home() / ".local" / "share" / "kurocode" / "prompt_history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    prompt_session = PromptSession(history=FileHistory(str(history_file)))
    session = Session(model_id=model)

    async with ConversationStore(ctx.config.db_path) as store:
        if resume:
            conv_id = resume
            msg_rows = await store.get_messages(conv_id)
            if not msg_rows:
                ctx.renderer.error(f"Conversation {resume} not found or empty.")
                sys.exit(1)
            for m in msg_rows:
                if m.role == "user":
                    session.add_user_message(m.content)
                elif m.role == "assistant":
                    session.add_assistant_message(m.content)
                elif m.role == "system":
                    session.add_system_message(m.content)
            ctx.renderer.info(f"Resumed conversation {conv_id} ({len(msg_rows)} messages)")
        else:
            conv_id = await store.create_conversation("Interactive Chat", session.model_id)
            ctx.renderer.info(f"Started new chat with {session.model_id}. Type /help for commands.")

        async with OpenRouterClient(ctx.config) as client:
            while True:
                try:
                    with patch_stdout():
                        user_input = await prompt_session.prompt_async("\n> ")
                except (EOFError, KeyboardInterrupt):
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await handle_slash_command(user_input, session, ctx)
                    continue

                session.add_user_message(user_input)
                await store.add_message(conv_id, "user", user_input)

                messages = session.to_openrouter_messages()
                assistant_response: list[str] = []

                try:
                    if ctx.no_stream:
                        resp = await client.chat(messages=messages, model=session.model_id)
                        content = resp.choices[0].message.content
                        assistant_response.append(content)
                        ctx.renderer.stream_token(content)
                        ctx.renderer.end_stream()
                    else:
                        async for chunk in client.stream_chat(messages=messages, model=session.model_id):
                            delta = chunk.choices[0].delta.content
                            if delta:
                                ctx.renderer.stream_token(delta)
                                assistant_response.append(delta)
                        ctx.renderer.end_stream()
                except KeyboardInterrupt:
                    ctx.renderer.end_stream()
                    ctx.renderer.info("\n[Stream interrupted. Saving partial response...]")
                
                full_response = "".join(assistant_response)
                if full_response:
                    session.add_assistant_message(full_response)
                    await store.add_message(conv_id, "assistant", full_response)


async def handle_slash_command(cmd: str, session: Session, ctx: CliContext) -> None:
    parts = cmd.split(maxsplit=1)
    command = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        ctx.renderer.info("Available commands:")
        ctx.renderer.info("  /help       - Show this help message")
        ctx.renderer.info("  /switch <m> - Switch the current model")
        ctx.renderer.info("  /model <m>  - Alias for /switch")
        ctx.renderer.info("  /clear      - Clear current session context")
        ctx.renderer.info("  /save <f>   - Save current session to a file")
        
    elif command in ("/switch", "/model"):
        if not arg:
            ctx.renderer.error("Usage: /switch <model_id>")
            return
        session.model_id = arg
        ctx.renderer.info(f"Switched model to: {arg}")
        
    elif command == "/clear":
        session.messages.clear()
        ctx.renderer.info("Session cleared.")
        
    elif command == "/save":
        if not arg:
            ctx.renderer.error("Usage: /save <filename>")
            return
        path = Path(arg)
        try:
            with path.open("w", encoding="utf-8") as f:
                for m in session.messages:
                    f.write(f"**{m.role.capitalize()}**:\n{m.content}\n\n")
            ctx.renderer.success(f"Saved session to {path}")
        except Exception as e:
            ctx.renderer.error(f"Failed to save session: {e}")
    else:
        ctx.renderer.error(f"Unknown command: {command}. Type /help for options.")
