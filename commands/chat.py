@click.command()
def hello():
    click.echo(click.style("Hi! I am free to use CLI tool", fg="green"))