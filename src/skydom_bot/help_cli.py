"""Top-level CLI help for the Skydom bot toolchain."""

from __future__ import annotations

import argparse
from textwrap import dedent


_TOPICS: dict[str, str] = {
    "inspect": dedent("""
        Inspect board geometry from a screenshot or live screen.

          skydom-inspect
          skydom-inspect --image .\\samples\\board.png
    """),
    "debug-vision": dedent("""
        Open the board-geometry visual debugger.

          skydom-debug-vision
          skydom-debug-vision --start-step 1
    """),
    "debug-tiles": dedent("""
        Open the interactive tile debugger.

          skydom-debug-tiles

        Useful keys:
          E  export selected cells to the dataset
          S  save the comparison image
          C  clear comparison selection
    """),
    "collect": dedent("""
        Capture the current board and collect every active cell.

          skydom-collect-tiles
          skydom-collect-tiles --image .\\samples\\board.png

        New crops are written directly to dataset/input/images/.
    """),
    "stats": dedent("""
        Summarize human-label coverage and validate canonical dataset records.

          skydom-dataset-stats
          skydom-dataset-stats --strict

        Bootstrap predictions under suggested are never counted as labels.
    """),
    "export": dedent("""
        Export newly collected samples as an incremental Label Studio batch.

          skydom-export-label-studio

        Then use Sync Storage in Label Studio.
    """),
    "import": dedent("""
        Import submitted Label Studio annotations back into dataset records.

          skydom-import-label-studio
    """),
    "label-studio": dedent("""
        Start the project-local Label Studio environment.

          .\\scripts\\start-label-studio.ps1

        Typical labeling cycle:
          skydom-collect-tiles
          skydom-export-label-studio
          # Sync + annotate in Label Studio
          skydom-import-label-studio
          skydom-dataset-stats
    """),
    "workflow": dedent("""
        Typical data-collection workflow:

          1. skydom-collect-tiles
          2. Continue playing and collect more boards as useful.
          3. skydom-export-label-studio
          4. Sync Source Storage in Label Studio.
          5. Annotate.
          6. skydom-import-label-studio
          7. skydom-dataset-stats

        For rare/suspicious pieces, use skydom-debug-tiles and press E to
        export only selected cells.
    """),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skydom",
        description="Command guide for the Skydom Reforged bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Commands:
              skydom-inspect              inspect board geometry
              skydom-debug-vision         visual board-geometry debugger
              skydom-debug-tiles          interactive tile debugger / selective export
              skydom-collect-tiles        capture and collect all active cells
              skydom-dataset-stats        summarize/validate human labels
              skydom-export-label-studio  create incremental Label Studio task batch
              skydom-import-label-studio  merge human labels back into records

            Quick workflow:
              skydom-collect-tiles
              skydom-export-label-studio
              # Sync + annotate in Label Studio
              skydom-import-label-studio
              skydom-dataset-stats

            More detail:
              skydom help workflow
              skydom help collect
              skydom help stats
              skydom help label-studio

            Every individual command also supports --help.
        """),
    )
    subparsers = parser.add_subparsers(dest="command")
    help_parser = subparsers.add_parser("help", help="Show help for a topic.")
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=sorted(_TOPICS),
        help="Topic to explain.",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    if args.command == "help":
        if args.topic is None:
            parser.print_help()
        else:
            print(_TOPICS[args.topic].rstrip())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
