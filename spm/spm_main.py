"""
DIVE-KT SPM Pipeline — Entry Point (v2)

三阶段流水线:
  阶段 0: subtitle_digest  —— 预计算字幕蒸馏缓存（独立，可跳过）
  阶段 1: problem          —— 题目语义先验构建
  阶段 2: video            —— 视频语义先验构建（可消费 digest 缓存）

用法示例:
  # 完整三阶段流水线（推荐）
  python run_spm.py --target pipeline \\
      --input raw_problem.json --output problem_processed.json \\
      --video-input raw_video.json --video-output video_processed.json \\
      --course course.json

  # 仅预计算字幕 digest
  python run_spm.py --target subtitle_digest \\
      --video-input raw_video.json --digest-output video_digests.jsonl

  # 单独跑 video（使用已有 digest 缓存）
  python run_spm.py --target video \\
      --video-input raw_video.json --video-output video_processed.json \\
      --course course.json --problems problem_processed.json \\
      --digest video_digests.jsonl
"""
import argparse
import json

from config import LLMConfig, SPMConfig, TRACE_DIR, logger
from llm_generator import VLLMTextGenerator
from processors import ProblemProcessor, SubtitleDigestPreprocessor, VideoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIVE-KT SPM")

    p.add_argument(
        "--target", type=str, required=True,
        choices=["subtitle_digest", "problem", "video", "digest2video", "pipeline"],
        help=(
            "subtitle_digest: 预计算字幕蒸馏缓存; "
            "problem: 题目处理; "
            "video: 视频处理; "
            "pipeline: 完整三阶段"
            "digest2video: 预计算字幕蒸馏缓存 & 视频处理; "
        ),
    )

    # 题目相关
    p.add_argument("--input",  type=str, default="", help="raw_problem.json 路径")
    p.add_argument("--output", type=str, default="", help="problem_processed.json 输出路径")

    # 视频相关
    p.add_argument("--video-input",  type=str, default="", help="raw_video.json 路径")
    p.add_argument("--video-output", type=str, default="", help="video_processed.json 输出路径")
    p.add_argument("--course",       type=str, default="", help="course.json 路径")
    p.add_argument("--problems",     type=str, default="",
                   help="已处理的 problem_processed.json 路径（视频模式可选）")

    # 字幕 digest 相关
    p.add_argument("--digest",        type=str, default="",
                   help="video_digests.jsonl 路径（video 模式读取；pipeline 模式自动推断）")
    p.add_argument("--digest-output", type=str, default="",
                   help="subtitle_digest 阶段的输出路径（默认与 --video-output 同目录）")

    # 保持向后兼容
    p.add_argument("--video-data-only", action=argparse.BooleanOptionalAction, default=True)

    # 批处理参数
    p.add_argument("--batch-size",    type=int, default=4)
    p.add_argument("--flush-every",   type=int, default=200)
    p.add_argument("--max-linked-problems", type=int, default=6)

    # 字幕预处理参数
    p.add_argument("--subtitle-max-segments", type=int, default=40,
                   help="分段抽样后最多保留的字幕片段数")
    p.add_argument("--subtitle-n-strata", type=int, default=5,
                   help="字幕分层抽样的层数（默认 5）")

    return p.parse_args()


# ── 各阶段入口 ────────────────────────────────────────────────

def run_subtitle_digest(args, generator) -> str:
    """
    阶段 0：预计算字幕蒸馏。
    返回 digest 输出路径（供后续阶段使用）。
    """
    video_input = args.video_input
    if not video_input:
        raise ValueError("subtitle_digest 需要 --video-input")

    # 默认输出路径：与 video-output 同目录，文件名 video_digests.jsonl
    if args.digest_output:
        digest_out = args.digest_output
    elif args.video_output:
        import os
        digest_out = os.path.join(
            os.path.dirname(args.video_output), "video_digests.jsonl"
        )
    else:
        digest_out = "video_digests.jsonl"

    logger.info(f"[subtitle_digest] 输入: {video_input} → 输出: {digest_out}")

    proc = SubtitleDigestPreprocessor(
        input_path=video_input,
        output_path=digest_out,
        generator=generator,
        subtitle_max_segments=args.subtitle_max_segments,
        subtitle_n_strata=args.subtitle_n_strata,
        trace_dir=TRACE_DIR,
    )
    proc.run(batch_size=args.batch_size, flush_every=args.flush_every)
    return digest_out


def run_problem(args, generator) -> None:
    if not args.input or not args.output:
        raise ValueError("problem 模式需要 --input 和 --output")
    proc = ProblemProcessor(
        input_path=args.input,
        output_path=args.output,
        generator=generator,
        trace_dir=TRACE_DIR,
    )
    proc.run(batch_size=args.batch_size, flush_every=args.flush_every)


def run_video(args, generator, problem_path: str = "", digest_path: str = "") -> None:
    video_input  = args.video_input or args.input
    video_output = args.video_output or args.output
    if not video_input or not video_output:
        raise ValueError("video 模式需要 --video-input 和 --video-output")
    if not args.course:
        raise ValueError("video 模式需要 --course")

    spm_config = SPMConfig(max_linked_problems=args.max_linked_problems)
    proc = VideoProcessor(
        input_path=video_input,
        output_path=video_output,
        generator=generator,
        course_file_path=args.course,
        problem_file_path=problem_path or args.problems,
        digest_file_path=digest_path or args.digest,
        video_data_only=args.video_data_only,
        spm_config=spm_config,
        trace_dir=TRACE_DIR,
    )
    proc.run(batch_size=args.batch_size, flush_every=args.flush_every)


# ── 主入口 ───────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info(f"SPM Pipeline: target={args.target}")
    logger.info("Project Configuration:\n" + json.dumps(vars(args), indent=4, ensure_ascii=False))

    generator = VLLMTextGenerator(LLMConfig())

    if args.target == "subtitle_digest":
        run_subtitle_digest(args, generator)

    elif args.target == "problem":
        run_problem(args, generator)

    elif args.target == "video":
        run_video(args, generator)

    elif args.target == "digest2video":
        logger.info("═══ 阶段 0: Subtitle Digest ═══")
        digest_path = ""
        if args.video_input:
            try:
                digest_path = run_subtitle_digest(args, generator)
            except Exception as e:
                logger.warning(f"阶段 0 跳过（{e}）；将在无字幕模式下处理视频")

        logger.info("═══ 阶段 1: Video ═══")
        run_video(
            args, generator,
            problem_path=args.output,
            digest_path=digest_path,
        )

    elif args.target == "pipeline":
        # 阶段 0：字幕 digest（有字幕则预计算，无字幕跳过不报错）
        logger.info("═══ 阶段 0: Subtitle Digest ═══")
        digest_path = ""
        if args.video_input:
            try:
                digest_path = run_subtitle_digest(args, generator)
            except Exception as e:
                logger.warning(f"阶段 0 跳过（{e}）；将在无字幕模式下处理视频")

        # 阶段 1：题目
        logger.info("═══ 阶段 1: Problem ═══")
        run_problem(args, generator)

        # 阶段 2：视频（使用阶段 0 的 digest + 阶段 1 的 problems）
        logger.info("═══ 阶段 2: Video ═══")
        run_video(
            args, generator,
            problem_path=args.output,
            digest_path=digest_path,
        )


if __name__ == "__main__":
    main()
