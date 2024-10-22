"""
Medusa Package
"""

# Nice outputs via rich
from rich.console import Console

console = Console()

# Units via pint
from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()  # TODO downselect the units
ureg.setup_matplotlib()  # support for plotting

# Define characteristic quantities as units
ureg.define("LU = 1 km")
ureg.define("TU = 1 sec")
ureg.define("MU = 1 kg")

set_application_registry(ureg)  # enables pickling and unpickling of Quantities


__all__ = ["corrections", "data", "dynamics", "numerics", "propagate", "plots", "util"]
