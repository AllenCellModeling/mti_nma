import inspect
import fire
from mti_nma import steps
from mti_nma.bin.all import All
from mti_nma.bin.quilt_init import QuiltInit


def cli():
    step_map = {
        name.lower(): step
        for name, step in inspect.getmembers(steps)
        if inspect.isclass(step)
    }

    fire.Fire({**step_map, "all": All, "quilt": QuiltInit})
