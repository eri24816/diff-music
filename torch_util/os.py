import subprocess
def run_command(command, verbose=False):
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = p.stdout
    stderr = p.stderr
    assert stdout
    assert stderr
    if verbose:
        while p.poll() is None:
            print(stdout.readline().decode("utf-8"), end="")
            print(stdout.readline().decode("utf-8"), end="")
        print(stdout.read().decode("utf-8"), end="")
        print(stderr.read().decode("utf-8"), end="")
    else:
        p.wait()
    assert p.returncode == 0, f"Command failed: {command}"
