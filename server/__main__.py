from __future__ import annotations

import uvicorn

from server.config import load_settings

if __name__ == "__main__":
    s = load_settings()
    uvicorn.run(
        "server.app:create_app",
        factory=True,
        host=s.host,
        port=s.port,
        log_level=s.log_level,
        workers=1,
    )
