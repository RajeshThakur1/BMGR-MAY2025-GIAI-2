from fastapi import WebSocket
from collections import defaultdict

# channel -> {username -> WebSocket}
subscribers: dict[str, dict[str, WebSocket]] = defaultdict(dict)
