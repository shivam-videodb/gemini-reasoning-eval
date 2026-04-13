"""Pipeline orchestrator for the gemini-reasoning-eval benchmark.

Handles: upload -> extract scenes -> infer (VLM) -> evaluate (metrics) -> visualize.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import get_api_key
from src.schemas import (
    SceneMetadata,
    SceneMetrics,
    VariantMetrics,
    AggregatedResults,
)

logger = logging.getLogger(__name__)


def _save_json(data: Any, path: Path) -> None:
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if hasattr(data, "model_dump"):
            json.dump(data.model_dump(), f, indent=2, default=str)
        elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
            json.dump([item.model_dump() for item in data], f, indent=2, default=str)
        else:
            json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {path}")


def _load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path) as f:
        return json.load(f)


class Pipeline:
    """Orchestrates the full evaluation pipeline."""

    def __init__(self, config: dict[str, Any], resume: bool = False):
        self.config = config
        self.resume = resume
        self.output_dir = Path(config["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Upload Videos -------------------------------------------------

    async def upload(self) -> dict[str, str]:
        """Upload videos to VideoDB and return a source -> media_id manifest."""
        import videodb
        from tqdm import tqdm

        api_key = get_api_key("VIDEO_DB_API_KEY")
        conn = videodb.connect(api_key=api_key)

        video_config = self.config["videos"]
        manifest_path = self.output_dir / "upload_manifest.json"

        # Check for existing collection
        if video_config.get("collection_id"):
            coll = conn.get_collection(video_config["collection_id"])
            videos = coll.get_videos()
            max_videos = video_config.get("max_videos")
            if max_videos and max_videos > 0:
                videos = videos[:max_videos]
            manifest = {v.name: v.id for v in videos}
            _save_json(manifest, manifest_path)
            logger.info(f"Using existing collection: {len(manifest)} videos")
            return manifest

        # Resume from existing manifest
        if self.resume and manifest_path.exists():
            manifest = _load_json(manifest_path)
            logger.info(f"Resumed upload manifest: {len(manifest)} videos")
            return manifest

        # Upload new videos to the default collection
        coll = conn.get_collection()

        sources = video_config.get("sources", [])
        manifest: dict[str, str] = {}

        for source in tqdm(sources, desc="Uploading videos"):
            if self.resume and source in manifest:
                continue
            try:
                source_path = Path(source)
                if source_path.exists() and source_path.is_file():
                    video = coll.upload(file_path=str(source_path))
                else:
                    video = coll.upload(url=source)
                manifest[source] = video.id
                logger.info(f"Uploaded: {source} -> {video.id}")
            except Exception as e:
                logger.error(f"Failed to upload {source}: {e}")
                manifest[source] = f"ERROR: {e}"

        _save_json(manifest, manifest_path)
        return manifest

    # -- Step 2: Extract Scenes ------------------------------------------------

    async def extract(self, manifest: dict[str, str] | None = None) -> dict[str, list[dict]]:
        """Extract scenes from uploaded videos using VideoDB."""
        import videodb
        from tqdm import tqdm

        api_key = get_api_key("VIDEO_DB_API_KEY")
        conn = videodb.connect(api_key=api_key)

        scenes_dir = self.output_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest if not provided
        if manifest is None:
            manifest_path = self.output_dir / "upload_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    "No upload manifest found. Run the upload step first."
                )
            manifest = _load_json(manifest_path)

        vdb_config = self.config.get("videodb", {})
        max_frames = vdb_config.get("max_frames_per_scene", 5)

        # Get collection for video access — fall back to default collection if none configured
        collection_id = self.config.get("videos", {}).get("collection_id")
        coll = conn.get_collection(collection_id) if collection_id else conn.get_collection()

        all_scenes: dict[str, list[dict]] = {}

        for source, media_id in tqdm(manifest.items(), desc="Extracting scenes"):
            if media_id.startswith("ERROR:"):
                continue

            scene_file = scenes_dir / f"{media_id}.json"

            # Resume: skip if scene file exists
            if self.resume and scene_file.exists():
                all_scenes[media_id] = _load_json(scene_file)
                continue

            try:
                video = coll.get_video(media_id)

                # Extract scenes with frame_count so frames come back with URLs attached.
                # If scenes already exist, fetch the existing collection instead.
                logger.info(f"Extracting scenes for {media_id} (this may take a minute)...")
                try:
                    scene_collection = video.extract_scenes(
                        extraction_config={"frame_count": max_frames}
                    )
                except Exception:
                    sc_list = video.list_scene_collection()
                    if sc_list:
                        sc_id = sc_list[0].get("scene_collection_id", "")
                        scene_collection = video.get_scene_collection(sc_id)
                    else:
                        scene_collection = None

                scenes = scene_collection.scenes if scene_collection else []
                logger.info(f"Processing {len(scenes)} scenes for {media_id}...")

                scene_list = []
                for idx, scene in enumerate(scenes, 1):
                    # Frames are already attached to each scene — no extra HTTP calls needed
                    frame_urls = [
                        f.url for f in (scene.frames or []) if getattr(f, "url", None)
                    ]

                    scene_data = {
                        "media_id": media_id,
                        "scene_idx": idx,
                        "scene_start": scene.start,
                        "scene_end": scene.end,
                        "num_frames": len(frame_urls),
                        "frame_urls": frame_urls,
                    }
                    scene_list.append(scene_data)

                all_scenes[media_id] = scene_list
                _save_json(scene_list, scene_file)
                logger.info(f"Extracted {len(scene_list)} scenes from {media_id}")

            except Exception as e:
                logger.error(f"Failed to extract scenes from {media_id}: {e}")

        total = sum(len(s) for s in all_scenes.values())
        logger.info(f"Total scenes extracted: {total} across {len(all_scenes)} videos")
        return all_scenes

    # -- Step 3: Run Inference -------------------------------------------------

    async def infer(self, all_scenes: dict[str, list[dict]] | None = None) -> None:
        """Run Gemini VLM inference on all scenes — parallel per scene, sequential per variant."""
        import io
        import asyncio

        import requests as http_requests
        from PIL import Image
        from tqdm import tqdm
        from src.gemini import GeminiVLM

        google_api_key = get_api_key("GOOGLE_API_KEY")
        gemini_config = self.config.get("gemini", {})
        prompt = gemini_config.get("prompt", "Describe this scene.")
        max_concurrent = gemini_config.get("max_concurrent", 10)

        # Load scenes if not provided
        if all_scenes is None:
            scenes_dir = self.output_dir / "scenes"
            if not scenes_dir.exists():
                raise FileNotFoundError("No scenes found. Run the extract step first.")
            all_scenes = {}
            for scene_file in sorted(scenes_dir.glob("*.json")):
                media_id = scene_file.stem
                all_scenes[media_id] = _load_json(scene_file)

        temperature = gemini_config.get("temperature", 0.0)

        # Support both new per-model variants spec and legacy flat lists
        variant_specs = gemini_config.get("variants")
        if variant_specs:
            model_budget_pairs = [
                (spec["model"], budget)
                for spec in variant_specs
                for budget in spec.get("thinking_budgets", [128])
            ]
        else:
            models = gemini_config.get("models", ["gemini-2.5-flash"])
            budgets = gemini_config.get("thinking_budgets", [128])
            model_budget_pairs = [
                (model, budget) for model in models for budget in budgets
            ]

        raw_dir = self.output_dir / "raw"

        async def load_frame(url: str) -> Any | None:
            """Load a single frame image in a thread to avoid blocking the event loop."""
            try:
                resp = await asyncio.to_thread(http_requests.get, url, timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content))
            except Exception as e:
                logger.warning(f"Failed to load frame {url}: {e}")
                return None

        for model_name, budget in model_budget_pairs:
            vlm = GeminiVLM(
                api_key=google_api_key,
                model=model_name,
                thinking_budget=budget,
                temperature=temperature,
            )
            variant_key = vlm._variant_key()
            variant_dir = raw_dir / variant_key
            variant_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Running inference: {variant_key} (max_concurrent={max_concurrent})")

            for media_id, scenes in tqdm(
                all_scenes.items(), desc=f"Inference: {variant_key}"
            ):
                output_file = variant_dir / f"{media_id}.json"

                # Resume: skip if output exists
                if self.resume and output_file.exists():
                    continue

                semaphore = asyncio.Semaphore(max_concurrent)

                async def process_scene(scene: dict) -> dict | None:
                    async with semaphore:
                        metadata = SceneMetadata(
                            media_id=media_id,
                            scene_idx=scene["scene_idx"],
                            scene_start=scene.get("scene_start", 0.0),
                            scene_end=scene.get("scene_end", 0.0),
                            num_frames=scene.get("num_frames", 0),
                        )

                        # Load all frames concurrently
                        frame_tasks = [
                            load_frame(url) for url in scene.get("frame_urls", [])
                        ]
                        frame_results = await asyncio.gather(*frame_tasks)
                        frames = [f for f in frame_results if f is not None]

                        if not frames:
                            logger.warning(
                                f"No frames for {media_id} scene {scene['scene_idx']}"
                            )
                            return None

                        result = await vlm.describe_scene(frames, prompt, metadata)
                        return result.model_dump()

                raw_results = await asyncio.gather(
                    *[process_scene(s) for s in scenes],
                    return_exceptions=True,
                )

                results = []
                for r in raw_results:
                    if isinstance(r, Exception):
                        logger.error(f"Scene failed: {r}")
                    elif r is not None:
                        results.append(r)

                _save_json(results, output_file)

        logger.info("Inference complete.")

    # -- Step 4: Evaluate Metrics ----------------------------------------------

    async def evaluate(self) -> None:
        """Compute evaluation metrics on raw VLM outputs — batches run in parallel."""
        import asyncio
        from tqdm import tqdm
        from src.judge import LLMJudge
        from src.metrics import MetricsCalculator

        openai_key = get_api_key("OPENAI_API_KEY")
        judge_config = self.config.get("judge", {})

        judge = LLMJudge(
            api_key=openai_key,
            model=judge_config.get("model", "gpt-4o-mini"),
            max_retries=judge_config.get("max_retries", 5),
            retry_min_wait=judge_config.get("retry_min_wait", 2),
            retry_max_wait=judge_config.get("retry_max_wait", 60),
            tokens_per_minute=judge_config.get("tokens_per_minute", 2_000_000),
            cache_dir=str(self.output_dir / "cache"),
        )

        calc = MetricsCalculator(
            judge=judge,
            fuzzy_threshold=judge_config.get("fuzzy_match_threshold", 75),
        )

        raw_dir = self.output_dir / "raw"
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            raise FileNotFoundError("No raw results found. Run the infer step first.")

        batch_size = judge_config.get("batch_size", 10)
        max_concurrent = judge_config.get("max_concurrent", 5)
        all_variant_metrics: list[VariantMetrics] = []
        total_scenes = 0
        video_ids: set[str] = set()

        for variant_dir in sorted(raw_dir.iterdir()):
            if not variant_dir.is_dir():
                continue

            variant_key = variant_dir.name
            metrics_file = metrics_dir / f"{variant_key}.json"

            # Parse model name and budget from variant key
            parts = variant_key.rsplit("_", 1)
            model_name = parts[0]
            budget = int(parts[1]) if len(parts) > 1 else 0

            # Resume: skip evaluation but still include in aggregation
            if self.resume and metrics_file.exists():
                existing = _load_json(metrics_file)
                total_scenes += len(existing)
                existing_metrics = [SceneMetrics(**s) for s in existing]
                all_variant_metrics.append(
                    self._aggregate_variant(existing_metrics, model_name, budget)
                )
                continue

            logger.info(f"Evaluating: {variant_key}")

            # Load all scene results for this variant
            all_scene_data: list[dict] = []
            for result_file in sorted(variant_dir.glob("*.json")):
                scenes = _load_json(result_file)
                for scene in scenes:
                    video_ids.add(scene.get("metadata", {}).get("media_id", ""))
                    all_scene_data.append(scene)

            # Build all batches
            batches = [
                all_scene_data[i : i + batch_size]
                for i in range(0, len(all_scene_data), batch_size)
            ]

            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_batch(batch: list[dict]) -> list[SceneMetrics]:
                async with semaphore:
                    return await self._evaluate_batch(batch, calc, model_name, budget)

            # Run all batches in parallel (bounded by semaphore)
            pbar = tqdm(total=len(batches), desc=f"Evaluate: {variant_key}")
            scene_metrics: list[SceneMetrics] = []

            batch_tasks = [run_batch(b) for b in batches]
            for coro in asyncio.as_completed(batch_tasks):
                batch_result = await coro
                scene_metrics.extend(batch_result)
                pbar.update(1)
            pbar.close()

            _save_json(scene_metrics, metrics_file)
            total_scenes += len(scene_metrics)

            # Compute variant-level aggregation
            variant_agg = self._aggregate_variant(scene_metrics, model_name, budget)
            all_variant_metrics.append(variant_agg)

        # Save aggregated results
        aggregated = AggregatedResults(
            variants=all_variant_metrics,
            video_count=len(video_ids),
            total_scenes=total_scenes,
        )
        _save_json(aggregated, metrics_dir / "aggregated.json")
        logger.info(
            f"Evaluation complete: {total_scenes} scenes, "
            f"{len(all_variant_metrics)} variants, "
            f"LLM judge calls: {judge.call_count}"
        )

    async def _evaluate_batch(
        self,
        batch: list[dict],
        calc: Any,
        model_name: str,
        budget: int,
    ) -> list[SceneMetrics]:
        """Evaluate a batch of scenes.

        Splits the batch based on actual thought content (not budget alone),
        since some budgets produce thought tokens but don't return thought text.
        """
        results: dict[int, SceneMetrics] = {}

        # Split by actual thought content
        thought_indices = [i for i, s in enumerate(batch) if s.get("thought")]
        no_thought_indices = [i for i, s in enumerate(batch) if not s.get("thought")]

        # Path A: Scenes WITH thought text -> coverage + dominant
        if thought_indices:
            coverage_data = []
            for i in thought_indices:
                scene = batch[i]
                coverage_data.append({
                    "thought_stream": scene["thought"],
                    "final_output": scene.get("response", {}),
                })

            coverage_results, _ = await calc.compute_coverage_batch(coverage_data)
            extractions = [r[0] for r in coverage_results]
            dominant_results, _ = await calc.compute_dominant_batch(extractions)

            for j, orig_idx in enumerate(thought_indices):
                scene = batch[orig_idx]
                thought = scene["thought"]
                extraction, tc, og = coverage_results[j]
                dominant = dominant_results[j]
                cf = calc.compute_contentfulness(thought)

                f1 = None
                if tc is not None and og is not None and (tc + og) > 0:
                    f1 = 2 * tc * og / (tc + og)

                tokens = scene.get("tokens", {})
                results[orig_idx] = SceneMetrics(
                    key=scene.get("key", ""),
                    model=model_name,
                    thinking_budget=budget,
                    contentfulness=cf,
                    thought_coverage=tc,
                    output_grounding=og,
                    f1=f1,
                    dominant_subject=dominant.main_subject,
                    dominant_action=dominant.main_action,
                    dominant_setting=dominant.main_setting,
                    thought_tokens=tokens.get("thought_tokens", 0),
                    input_tokens=tokens.get("input_tokens", 0),
                    output_tokens=tokens.get("output_tokens", 0),
                    image_tokens=tokens.get("image_tokens", 0),
                    text_tokens=tokens.get("text_tokens", 0),
                    total_tokens=tokens.get("total_tokens", 0),
                )

        # Path B: Scenes WITHOUT thought text -> final-only extraction
        if no_thought_indices:
            final_data = [{"final_output": batch[i].get("response", {})} for i in no_thought_indices]
            final_results, _ = await calc.compute_final_only_batch(final_data)

            for j, orig_idx in enumerate(no_thought_indices):
                scene = batch[orig_idx]
                ext = final_results[j]
                tokens = scene.get("tokens", {})
                results[orig_idx] = SceneMetrics(
                    key=scene.get("key", ""),
                    model=model_name,
                    thinking_budget=budget,
                    dominant_subject=ext.main_subject,
                    dominant_action=ext.main_action,
                    dominant_setting=ext.main_setting,
                    thought_tokens=tokens.get("thought_tokens", 0),
                    input_tokens=tokens.get("input_tokens", 0),
                    output_tokens=tokens.get("output_tokens", 0),
                    image_tokens=tokens.get("image_tokens", 0),
                    text_tokens=tokens.get("text_tokens", 0),
                    total_tokens=tokens.get("total_tokens", 0),
                )

        return [results[i] for i in range(len(batch)) if i in results]

    def _aggregate_variant(
        self,
        metrics: list[SceneMetrics],
        model_name: str,
        budget: int,
    ) -> VariantMetrics:
        """Compute aggregate statistics for a variant."""
        n = len(metrics)
        if n == 0:
            return VariantMetrics(model=model_name, thinking_budget=budget)

        cf_vals = [m.contentfulness.contentfulness for m in metrics if m.contentfulness]
        tc_vals = [m.thought_coverage for m in metrics if m.thought_coverage is not None]
        og_vals = [m.output_grounding for m in metrics if m.output_grounding is not None]
        f1_vals = [m.f1 for m in metrics if m.f1 is not None]
        thought_tokens = [m.thought_tokens for m in metrics]
        input_tokens = [m.input_tokens for m in metrics]
        output_tokens = [m.output_tokens for m in metrics]
        image_tokens = [m.image_tokens for m in metrics]
        text_tokens = [m.text_tokens for m in metrics]
        total_tokens = [m.total_tokens for m in metrics]
        errors = sum(1 for m in metrics if m.key == "")

        import statistics

        def safe_mean(vals: list[float]) -> float:
            return statistics.mean(vals) if vals else 0.0

        def safe_std(vals: list[float]) -> float:
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        f1_mean_val = safe_mean(f1_vals) if f1_vals else None
        f1_std_val = safe_std(f1_vals) if len(f1_vals) > 1 else None
        f1_cv = (f1_std_val / f1_mean_val) if (f1_std_val is not None and f1_mean_val) else None

        return VariantMetrics(
            model=model_name,
            thinking_budget=budget,
            scene_count=n,
            contentfulness_mean=safe_mean(cf_vals),
            contentfulness_std=safe_std(cf_vals),
            thought_coverage_mean=safe_mean(tc_vals) if tc_vals else None,
            output_grounding_mean=safe_mean(og_vals) if og_vals else None,
            f1_mean=f1_mean_val,
            f1_std=f1_std_val,
            f1_cv=f1_cv,
            perfect_f1_pct=(sum(1 for v in f1_vals if v == 1.0) / len(f1_vals) * 100) if f1_vals else None,
            low_f1_pct=(sum(1 for v in f1_vals if v < 0.5) / len(f1_vals) * 100) if f1_vals else None,
            thought_tokens_mean=safe_mean(thought_tokens),
            input_tokens_mean=safe_mean(input_tokens),
            output_tokens_mean=safe_mean(output_tokens),
            image_tokens_mean=safe_mean(image_tokens),
            text_tokens_mean=safe_mean(text_tokens),
            total_tokens_mean=safe_mean(total_tokens),
            error_count=errors,
            error_pct=(errors / n * 100) if n > 0 else None,
        )

    # -- Step 5: Visualize -----------------------------------------------------

    def visualize(self) -> None:
        """Generate publication-quality figures from aggregated results."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("enabled", True):
            logger.info("Visualization disabled in config.")
            return

        metrics_dir = self.output_dir / "metrics"
        aggregated_path = metrics_dir / "aggregated.json"

        if not aggregated_path.exists():
            raise FileNotFoundError(
                "No aggregated results found. Run the evaluate step first."
            )

        from src.plots import generate_figures

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        aggregated = _load_json(aggregated_path)
        fmt = viz_config.get("format", "png")
        dpi = viz_config.get("dpi", 300)

        generate_figures(aggregated, str(figures_dir), fmt=fmt, dpi=dpi)
        logger.info(f"Figures saved to: {figures_dir}")

    # -- Run All Steps ---------------------------------------------------------

    async def run_all(self) -> None:
        """Run the full pipeline: upload -> extract -> infer -> evaluate -> visualize."""
        start = time.monotonic()

        logger.info("Step 1/5: Uploading videos...")
        manifest = await self.upload()

        logger.info("Step 2/5: Extracting scenes...")
        all_scenes = await self.extract(manifest)

        logger.info("Step 3/5: Running VLM inference...")
        await self.infer(all_scenes)

        logger.info("Step 4/5: Computing evaluation metrics...")
        await self.evaluate()

        logger.info("Step 5/5: Generating figures...")
        self.visualize()

        elapsed = time.monotonic() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
