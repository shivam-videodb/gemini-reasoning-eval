"""Entry point for the gemini-reasoning-eval pipeline.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/default.yaml --step upload
    python run.py --config configs/default.yaml --step evaluate --resume
"""

import argparse
import asyncio
import logging
import sys

from src.config import load_config
from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning quality in Gemini VLMs for video scene understanding.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/default.yaml).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "upload", "extract", "infer", "evaluate", "visualize"],
        help="Pipeline step to run (default: all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs, skipping completed steps.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    pipeline = Pipeline(config=config, resume=args.resume)

    step = args.step

    if step == "all":
        await pipeline.run_all()
    elif step == "upload":
        await pipeline.upload()
    elif step == "extract":
        await pipeline.extract()
    elif step == "infer":
        await pipeline.infer()
    elif step == "evaluate":
        await pipeline.evaluate()
    elif step == "visualize":
        pipeline.visualize()

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
