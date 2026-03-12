import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configures the global logging format.
    """
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        stream=sys.stdout
    )
