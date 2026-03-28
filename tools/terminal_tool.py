from langchain_core.tools import tool


@tool
def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess

    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return run_result.stdout if run_result.returncode == 0 else run_result.stderr


__all__ = ["run_terminal_command"]
