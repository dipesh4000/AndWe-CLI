import click
from commands import chat

@click.group()
@click.version_option()
def cli():
    pass


cli.add_command(chat.hello)