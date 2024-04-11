import subprocess


def run_cmd(cmd: str, return_stdout: bool = False):
    res = subprocess.run(cmd, shell=True, encoding="utf-8", capture_output=True)
    if res.returncode != 0:
        print(f"Error running command: {cmd}")
        print(res.stdout)
        print(res.stderr)
    else:
        lines = res.stdout.strip().splitlines()
        if len(lines) > 0:
            print(lines[0].strip())
            print("...")
            print(lines[-1].strip())
    assert res.returncode == 0, f"Command {cmd} failed with return code {res.returncode}"
    if return_stdout:
        return res.stdout
