from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import List, Dict
import os
from pathlib import Path

load_dotenv()

PORT = int(os.environ.get("PORT", 10000))
BASE_DIR = os.environ.get("BASE_DIR", "./shared")  # default local folder

# Normalize base directory path
BASE_DIR = os.path.abspath(BASE_DIR)
if not os.path.exists(BASE_DIR):
    raise Exception(f"BASE_DIR does not exist: {BASE_DIR}")

mcp = FastMCP("file-server",host="0.0.0.0", port=8000)


def safe_path(relative_path: str) -> str:
    """
    Safely join and validate the path to prevent path traversal.
    """
    resolved = Path(BASE_DIR, relative_path).resolve()
    if not str(resolved).startswith(BASE_DIR):
        raise Exception("Access denied: Path traversal attempt detected")
    return str(resolved)


@mcp.tool()
def list_files() -> List[Dict]:
    """
    List files and folders under the base directory.

    Args:
        None.

    Returns:
        A list of files and directories.
    """
    prefix="Users/jsebastian1/Library/CloudStorage/OneDrive-InfobloxInc/Boomi Projects/Chatbot/MCP_SERVER/shared"
    target_dir = safe_path(prefix)
    if not os.path.exists(target_dir):
        return [{"error": f"{prefix} does not exist"}]

    items = []
    for entry in os.listdir(target_dir):
        full_path = os.path.join(target_dir, entry)
        items.append({
            "name": entry,
            "path": os.path.relpath(full_path, BASE_DIR),
            "type": "folder" if os.path.isdir(full_path) else "file",
            "size": os.path.getsize(full_path) if os.path.isfile(full_path) else 0
        })
    return items


@mcp.tool()
def read_log_file() -> Dict:
    """
    Read the content of log files file inside the base directory.

    Args:
        None.

    Returns:
        The content of the file as text.
    """
    act_path='/Users/jsebastian1/Library/CloudStorage/OneDrive-InfobloxInc/Boomi Projects/Chatbot/MCP_SERVER/shared/Log.txt'
    file_path = act_path

    if not os.path.exists(file_path):
        return {"error": "File not found"}
    if not os.path.isfile(file_path):
        return {"error": "This path is not a file"}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return {
        "path": file_path,
        "content": content
    }




if __name__ == "__main__":
    print(f"✅ MCP File Server running on port {PORT}, base dir: {BASE_DIR}")
    mcp.run(transport="streamable-http")
