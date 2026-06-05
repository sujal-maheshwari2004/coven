import asyncio
import logging
import sys

from coven import Coven

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

    coven = Coven(model=model)
    dag  = await coven.run(task)
    print(coven.to_text(dag))


if __name__ == "__main__":
    asyncio.run(main())