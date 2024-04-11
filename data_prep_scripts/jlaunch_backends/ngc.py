import click
from .shared_utils import run_cmd


@click.command()
@click.argument("command", type=str)
@click.option("--instance", type=str, default="dgx1v.16g.1.norm", help="Instance type")
@click.option("--team", type=str, required=True, help="Team name")
@click.option("--name", type=str, required=True, help="Name of the environment")
@click.option(
    "--image",
    type=str,
    required=True,
    help="Docker image to use",
)
@click.option(
    "--result",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, writable=True),
    default="/result",
    help="Path to store results (e.g., /result)",
)
@click.option(
    "--workspace",
    type=(str, str),
    multiple=True,
    help="Workspace mount points. Can be used multiple times (e.g., user:host_path)",
)
def backend_ngc(
    command: str,
    instance: str,
    team: str,
    name: str,
    image: str,
    result: str,
    workspace: list[tuple[str, str]],
):
    """Launches a development environment with the specified options."""

    command_content = f"""ngc batch run --instance {instance} --team {team} --name {name} --image {image} --result {result}"""
    for user, host_path in workspace:
        command_content += f" --workspace {user}:{host_path}"
    command_content += f' --commandline "/workspace/entrypoint.sh; cd /project; echo STARTING USER CODE; {command}"'

    print(command_content)


if __name__ == "__main__":
    backend_ngc()
