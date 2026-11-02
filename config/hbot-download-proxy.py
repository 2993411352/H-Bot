#!/usr/bin/env python3
"""Loopback-only TCP relay to the existing HTTP proxy; shared download cap.

Only new HoloBrain jobs use port 27892. Asset downloads keep using 27891.
TLS stays end-to-end: this relay never decrypts or changes HTTP requests.
"""
import asyncio
import contextlib
import time

RATE = 2 * 1024 * 1024


class Limiter:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.next_slot = 0.0

    async def wait(self, count):
        async with self.lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + count / RATE
        await asyncio.sleep(max(0.0, start - now))


async def main():
    limiter = Limiter()

    async def relay(reader, writer, limited=False):
        try:
            while data := await reader.read(16384):
                if limited:
                    await limiter.wait(len(data))
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def handle(reader, writer):
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 27891), timeout=10
            )
            tasks = [
                asyncio.create_task(relay(reader, upstream_writer)),
                asyncio.create_task(relay(upstream_reader, writer, True)),
            ]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            print(f"relay connection failed: {type(exc).__name__}", flush=True)
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 27892)
    print("READY 127.0.0.1:27892 -> 127.0.0.1:27891; total cap 2 MiB/s", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
