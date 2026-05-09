import asyncio
import logging
import sys

from loom import Loom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run main.py \"<your task here>\"")
        sys.exit(1)

    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"

    loom = Loom(model=model)
    dag  = await loom.run(task)
    print(loom.to_text(dag))


if __name__ == "__main__":
    asyncio.run(main())