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
    "capture-screen": dedent("""
        Capture one manually triggered full game-area screenshot into the local corpus.

          skydom-capture-screen --configure
          skydom-capture-screen level-27 --meta mode=normal --meta goal=carrot

        The configured region excludes browser chrome and is reused on later captures.
        Images and metadata stay under corpus/ and are ignored by Git.
    """),
    "capture-gui": dedent("""
        Open a narrow persistent form for repeated corpus captures.

          skydom-capture-gui

        Fixed metadata fields: mode, initial_moves, board_variant, has_ice.
        Four extra name/value metadata rows are also available. Form values
        persist between captures and are restored on the next launch.
    """),
    "analyze-corpus": dedent("""
        Run the current BoardDetector against every captured corpus screen.

          skydom-analyze-corpus
          skydom-analyze-corpus --export-cells

        Writes corpus/derived/analysis.md, .csv and .jsonl with OK/REVIEW/FAIL
        status plus a review queue for unusual or failing scenarios.
    """),
    "collect": dedent("""
        Capture the current board and collect every active cell.

          skydom-collect-tiles
          skydom-collect-tiles --image .\\samples\\board.png

        New crops are written directly to dataset/input/images/.
    """),
    "stats": dedent("""
        Summarize dataset health and generate a training-readiness report.

          skydom-dataset-stats
          skydom-dataset-stats --details
          skydom-dataset-stats --strict

        Console output stays compact. By default a Markdown report is written
        to dataset/dataset-stats.md with collection guidance, coverage bands,
        capture-group provenance, and the recommended first-training floors.

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
              skydom-capture-screen       save one full game-area corpus screenshot
              skydom-capture-gui         narrow persistent capture form
              skydom-analyze-corpus       evaluate all corpus screenshots offline
              skydom-collect-tiles        capture and collect all active cells
              skydom-dataset-stats        summarize/validate human labels
              skydom-export-label-studio  create incremental Label Studio task batch
              skydom-import-label-studio  merge human labels back into records

            Screen-corpus workflow:
              skydom-capture-screen --configure
              skydom-capture-gui
              # or: skydom-capture-screen level-27 --meta mode=normal
              skydom-analyze-corpus

            Tile-labeling workflow:
              skydom-collect-tiles
              skydom-export-label-studio
              # Sync + annotate in Label Studio
              skydom-import-label-studio
              skydom-dataset-stats

            More detail:
              skydom help capture-screen
              skydom help capture-gui
              skydom help analyze-corpus
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
