"""
Cleanup Service for temporary audio files.

Manages TTL-based cleanup of temporary files with periodic background task.
Prevents disk-full outages from accumulated temporary audio files.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CleanupService:
    """Manages cleanup of temporary audio files with TTL-based expiration."""

    def __init__(
        self,
        uploads_dir: Path,
        ttl_hours: int = 1,
        cleanup_interval_minutes: int = 15,
    ):
        """Initialize the cleanup service.
        
        Args:
            uploads_dir: Path to the uploads directory containing temp files.
            ttl_hours: Time-to-live for temp files in hours (default: 1 hour).
            cleanup_interval_minutes: How often to run cleanup task (default: 15 min).
        """
        self.uploads_dir = uploads_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._running:
            logger.warning("CleanupService already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "CleanupService started",
            extra={
                "uploads_dir": str(self.uploads_dir),
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "cleanup_interval_minutes": self.cleanup_interval.total_seconds() / 60,
            },
        )

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        if not self._running:
            return

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("CleanupService stopped")

    async def _cleanup_loop(self) -> None:
        """Background loop that periodically runs cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                if self._running:
                    await self.cleanup_stale_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Cleanup loop error",
                    extra={"error": str(e)},
                    exc_info=True,
                )

    async def cleanup_stale_files(self) -> int:
        """Remove files older than TTL. Thread-safe for async.
        
        Returns:
            int: Number of files deleted.
        """
        if not self.uploads_dir.exists():
            logger.debug(f"Uploads directory does not exist: {self.uploads_dir}")
            return 0

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cleanup_stale_sync)

    def _cleanup_stale_sync(self) -> int:
        """Synchronous cleanup for stale files (runs in thread pool).
        
        Returns:
            int: Number of files deleted.
        """
        now = datetime.now()
        deleted = 0
        errors = 0

        try:
            for file_path in self.uploads_dir.glob("*"):
                if not file_path.is_file():
                    continue

                try:
                    file_stat = file_path.stat()
                    file_age = now - datetime.fromtimestamp(file_stat.st_mtime)

                    if file_age > self.ttl:
                        file_path.unlink()
                        deleted += 1
                        logger.debug(
                            "Deleted stale temporary file",
                            extra={
                                "file_path": file_path.name,
                                "age_hours": file_age.total_seconds() / 3600,
                            },
                        )
                except OSError as e:
                    logger.warning(
                        "Failed to delete file",
                        extra={"file_path": str(file_path), "error": str(e)},
                    )
                    errors += 1

        except Exception as e:
            logger.error(
                "Cleanup stale files failed",
                extra={"uploads_dir": str(self.uploads_dir), "error": str(e)},
                exc_info=True,
            )

        # Log summary
        if deleted > 0 or errors > 0:
            try:
                total_size_mb = sum(
                    f.stat().st_size for f in self.uploads_dir.glob("*") if f.is_file()
                ) / (1024**2)
            except Exception:
                total_size_mb = 0.0

            logger.info(
                "Cleanup task complete",
                extra={
                    "files_deleted": deleted,
                    "errors": errors,
                    "remaining_dir_size_mb": total_size_mb,
                },
            )

        return deleted

    async def cleanup_file(self, file_path: Path) -> bool:
        """Immediately delete a temporary file.
        
        Args:
            file_path: Path to the file to delete.
        
        Returns:
            bool: True if successfully deleted, False otherwise.
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.warning(
                "Failed to cleanup file",
                extra={"file_path": str(file_path), "error": str(e)},
            )
            return False

    async def get_dir_stats(self) -> dict:
        """Get statistics about the uploads directory.
        
        Returns:
            dict: Contains file_count, total_size_mb, oldest_file_age_hours.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_dir_stats_sync)

    def _get_dir_stats_sync(self) -> dict:
        """Synchronous directory stats (runs in thread pool)."""
        if not self.uploads_dir.exists():
            return {
                "file_count": 0,
                "total_size_mb": 0.0,
                "oldest_file_age_hours": 0.0,
            }

        now = datetime.now()
        file_count = 0
        total_size = 0
        oldest_time = now

        try:
            for file_path in self.uploads_dir.glob("*"):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < oldest_time:
                        oldest_time = file_mtime
        except Exception as e:
            logger.warning(f"Error calculating dir stats: {e}")

        oldest_age = (now - oldest_time).total_seconds() / 3600

        return {
            "file_count": file_count,
            "total_size_mb": total_size / (1024**2),
            "oldest_file_age_hours": oldest_age,
        }
