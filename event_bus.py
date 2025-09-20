import asyncio
import logging
from typing import Dict, Set, Any
from uuid import UUID
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[UUID, Set[asyncio.Queue[Dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def subscribe(self, user_id: UUID) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        
        async with self._lock:
            self._subscribers[user_id].add(queue)
        
        logger.info(f"New subscription for user {user_id}. Total: {len(self._subscribers[user_id])}")
        return queue
    
    async def unsubscribe(self, user_id: UUID, queue: asyncio.Queue[Dict[str, Any]]):
        async with self._lock:
            self._subscribers[user_id].discard(queue)
            if not self._subscribers[user_id]:
                del self._subscribers[user_id]
        
        logger.info(f"Unsubscribed user {user_id}. Remaining: {len(self._subscribers.get(user_id, []))}")
    
    async def publish(self, user_id: UUID, event_data: Dict[str, Any]):
        async with self._lock:
            subscribers = self._subscribers.get(user_id, set()).copy()
        
        if not subscribers:
            logger.debug(f"No subscribers for user {user_id}")
            return
        
        logger.info(f"Publishing event to {len(subscribers)} subscribers for user {user_id}")
        
        for queue in subscribers.copy():
            try:
                queue.put_nowait(event_data)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for user {user_id}, dropping event")
                async with self._lock:
                    self._subscribers[user_id].discard(queue)
    
    def get_subscriber_count(self, user_id: UUID) -> int:
        return len(self._subscribers.get(user_id, set()))

event_bus = EventBus()
