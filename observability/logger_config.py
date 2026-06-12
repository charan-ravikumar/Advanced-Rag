# ============================================
# IMPORTS
# ============================================

import logging


# ============================================
# LOGGER CONFIG
# ============================================

logging.basicConfig(

    level=logging.INFO,

    format=(

        "%(asctime)s | "

        "%(levelname)s | "

        "%(name)s | "

        "%(message)s"
    )
)


# ============================================
# LOGGER FACTORY
# ============================================

def get_logger(
    name
):

    return logging.getLogger(name)

logging.getLogger(
    "opentelemetry"
).setLevel(logging.ERROR)
