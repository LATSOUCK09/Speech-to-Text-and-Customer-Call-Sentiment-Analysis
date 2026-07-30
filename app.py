import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gradio_demo import create_demo

app = create_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        quiet=True,
        show_error=True,
    )
