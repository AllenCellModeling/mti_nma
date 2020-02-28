import inspect
import fire
from mti_nma import steps
from mti_nma.bin.all import All
from mti_nma.bin.quilt_init import QuiltInit
import logging
from unittest import mock

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


def cli():
    step_map = {
        name.lower(): step
        for name, step in inspect.getmembers(steps)
        if inspect.isclass(step)
    }

    # Interrupt fire print return
    with mock.patch("fire.core._PrintResult"):
        fire.Fire({**step_map, "all": All, "quilt": QuiltInit})
