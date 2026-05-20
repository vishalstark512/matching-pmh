"""pmh-train: thin CLI — try, doctor, evaluate (Step 5), route (+ optional wizard)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_route(args: argparse.Namespace) -> int:
    if getattr(args, "wizard", False):
        from pmh.onboarding import run_wizard

        if args.non_interactive and args.stack is None and args.task is None:
            print("route --wizard --non-interactive requires --stack or --task", file=sys.stderr)
            return 2
        run_wizard(
            stack=args.stack,
            task_id=args.task,
            has_target_domain=args.target_domain,
            has_target_labels=args.target_labels,
            has_frozen_features=args.frozen_features,
            has_style_pairs=args.style_pairs,
            interactive=not args.non_interactive,
        )
        return 0

    from pmh.task_router import explain_task, format_search_results, format_task_menu

    if getattr(args, "search", None):
        print(format_search_results(args.search))
        return 0
    if args.list:
        print(format_task_menu(short=False))
        return 0
    if args.task is None:
        print(format_task_menu(short=True))
        from pmh.adoption import format_recipe_banner

        print()
        print(format_recipe_banner())
        print("\nExample: pmh-train try --quick")
        print("         pmh-train route --task pose_or_keypoints")
        print("         pmh-train route --wizard --stack pytorch")
        print("Docs:    docs/START.md")
        return 0
    try:
        print(explain_task(args.task))
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2
    if not args.quiet:
        print("\nSetup snippet: pmh-train route --wizard --task", args.task)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from pmh.doctor import run_doctor

    artifact = str(args.artifact) if getattr(args, "artifact", None) else None
    rep = run_doctor(
        stack=args.stack,
        artifact_path=artifact,
        rank=getattr(args, "rank", None),
    )
    print(rep.summary())
    return 0 if rep.ok else 1


def main(argv: list[str] | None = None) -> int:
    from pmh.adoption import RECIPE_ONE_LINER

    parser = argparse.ArgumentParser(
        prog="pmh-train",
        description=f"Matched PMH — {RECIPE_ONE_LINER}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser(
        "route",
        help="Pick your ML task and get steps + doc links (optional setup wizard)",
    )
    p_route.add_argument(
        "--task",
        default=None,
        help="Task id, e.g. pose_or_keypoints, vision_classification",
    )
    p_route.add_argument(
        "--search",
        default=None,
        metavar="KEYWORD",
        help="Find applications by keyword (pose, hospital, blur, temporal, …)",
    )
    p_route.add_argument("--list", action="store_true", help="Print task menu")
    p_route.add_argument("--quiet", action="store_true", help="No follow-up hints")
    p_route.add_argument(
        "--wizard",
        action="store_true",
        help="Run interactive setup guide (stack, snippet) instead of printing route only",
    )
    p_route.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default=None,
        help="With --wizard: skip questionnaire when set",
    )
    p_route.add_argument("--target-domain", action="store_true", default=None)
    p_route.add_argument("--no-target-domain", action="store_false", dest="target_domain")
    p_route.add_argument("--target-labels", action="store_true", default=None)
    p_route.add_argument("--frozen-features", action="store_true", default=None)
    p_route.add_argument("--style-pairs", action="store_true", default=None)
    p_route.add_argument(
        "--non-interactive",
        action="store_true",
        help="With --wizard: use flags only (requires --stack or --task)",
    )
    p_route.set_defaults(func=_cmd_route, target_domain=True)

    p_doc = sub.add_parser("doctor", help="Check install and optional extras")
    p_doc.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default="pytorch",
    )
    p_doc.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional saved SigmaTaskEstimate — report preflight",
    )
    p_doc.add_argument("--rank", type=int, default=None)
    p_doc.set_defaults(func=_cmd_doctor)

    from pmh.cli.evaluate import add_evaluate_parser
    from pmh.cli.try_cmd import add_try_parser

    add_try_parser(sub)
    add_evaluate_parser(sub)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
