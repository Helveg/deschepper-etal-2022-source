def network_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "networks", *args
    )

def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )
