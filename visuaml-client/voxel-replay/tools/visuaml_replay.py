"""Source-tree entry point. Installed users can run visuaml-replay."""
from visuaml.replay.cli import main

if __name__ == "__main__":
    main()
else:
    from visuaml.replay import Replay, make_bundle  # compatibility with archived scripts

    __all__ = ["Replay", "make_bundle", "main"]
