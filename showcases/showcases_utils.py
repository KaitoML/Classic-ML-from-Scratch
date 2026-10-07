def prepare_notebook():
    """
    Adds the project root to sys.path so that `ml_code` can be imported
    from notebooks living in the `showcases/` folder.
    """
    import sys
    from pathlib import Path

    ROOT = Path.cwd()
    if ROOT.name == "showcases":
        ROOT = ROOT.parent

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))