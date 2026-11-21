"""Source-tree entry point. Installed users can run visuaml-replay."""
from visuaml.replay import Replay, make_bundle  # re-exported for archived scripts
from visuaml.replay.cli import main

__all__ = ["Replay", "make_bundle", "main"]

if __name__ == "__main__":
    main()
