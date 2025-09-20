import json
from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import StreamingResponse
from uuid import UUID
from typing import AsyncGenerator, Any, Dict
import asyncio
import logging
from event_bus import EventBus

logger = logging.getLogger(__name__)

class StreamingService:
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def create_user_stream(self, user_id: UUID, request: Request) -> StreamingResponse:
        """Create SSE stream for a specific user"""
        
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events for the user"""
            queue: asyncio.Queue[Dict[str, Any]] | None = None
            try:
                queue = await self.event_bus.subscribe(user_id)
                
                logger.info(f"Started stream for user {user_id}")
                
                yield self._format_sse_event({
                    "type": "connection",
                    "message": "Stream connected",
                    "user_id": str(user_id),
                    "timestamp": datetime.now(timezone.utc).isoformat() + 'Z'
                })
                
                while True:
                    try:
                        if await request.is_disconnected():
                            logger.info(f"Client disconnected for user {user_id}")
                            break
                        
                        try:
                            event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                            yield self._format_sse_event(event_data)
                        except asyncio.TimeoutError:
                            yield self._format_sse_heartbeat()
                    
                    except Exception as e:
                        logger.error(f"Error in event stream for user {user_id}: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Failed to create stream for user {user_id}: {e}")
            finally:
                if queue:
                    await self.event_bus.unsubscribe(user_id, queue)
                logger.info(f"Closed stream for user {user_id}")
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    def _format_sse_event(self, data: dict) -> str:
        """Format data as SSE event"""
        json_data = json.dumps(data, ensure_ascii=False)
        return f"data: {json_data}\n\n"
    
    def _format_sse_heartbeat(self) -> str:
        """Format heartbeat event"""
        return f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'})}\n\n"
