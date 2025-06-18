#!/usr/bin/env sh
""":"
# Polyglot trick: Python sees a triple-quoted string, bash sees "" then ":" (no-op)
# Install UV if needed (zero-dependency setup) – redirect output to stderr to preserve clean stdout
which uv >/dev/null || {
    echo ">>> uv install required" >&2 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh >&2 \
    && echo ">>> uv install done\n\n" >&2
}

# Execute with UV, replacing shell process (preserves stdio)
exec uv run --script --quiet "$0" "$@"
":"""


# /// script
# requires-python = ">=3.12"
# dependencies = ["langflow"]
# [tool.uv.sources]
# langflow = { git = "https://github.com/langflow-ai/langflow.git", rev = "main" }
# ///


import sys, json, pkgutil, importlib, asyncio

# Usage check
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} /path/to/flow.json \"Your chat prompt here...\"", file=sys.stderr)
    sys.exit(1)

json_path = sys.argv[1]
chat_prompt = sys.argv[2]

# --- bootstrap your DB models ---
import langflow.services.database.models  # noqa: F401
models_pkg = langflow.services.database.models
for _, module_name, _ in pkgutil.walk_packages(models_pkg.__path__, models_pkg.__name__ + "."):
    importlib.import_module(module_name)

from langflow.services.deps import get_settings_service
from langflow.services.database.service import DatabaseService

settings = get_settings_service()
settings.settings.database_url = "sqlite+aiosqlite:///:memory:"
db = DatabaseService(settings_service=settings)
asyncio.run(db.create_db_and_tables())

# --- load the flow definition ---
with open(json_path) as f:
    obj = json.load(f)

flow_id, flow_name, payload = obj["id"], obj.get("name", ""), obj["data"]

from langflow.processing.process import process_tweaks, run_graph_internal
from langflow.graph.graph.base import Graph
from langflow.api.v1.schemas import InputValueRequest

# --- build the graph ---
tweaked = process_tweaks(payload, {})
graph = Graph.from_payload(
    tweaked,
    flow_id=str(flow_id),
    user_id="none",
    flow_name=flow_name,
)

input_ids  = [v.id for v in graph.vertices if v.is_input]
output_ids = [v.id for v in graph.vertices if v.is_output]

# --- prepare your chat input from the CLI ---
inputs = [
    InputValueRequest(
        components=[input_ids[0]],
        input_value=chat_prompt,
        type="chat",
    )
]

# --- run it! ---
async def main():
    results, session_id = await run_graph_internal(
        graph=graph,
        flow_id=str(flow_id),
        session_id=None,
        inputs=inputs,
        outputs=output_ids,
        stream=False,
    )
    for r in results:
        for output in r.outputs:
            print(f"Flow result > {output.results['message'].text}")

if __name__ == "__main__":
    asyncio.run(main())

