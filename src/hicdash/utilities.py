"""Very basic package-wide utility functions for e.g. simple unit conversions.
"""


def chr_unprefix(chr_string: str) -> str:
    """Remove the "chr" prefix from a chromosome string.

    If no "chr" prefix is present, returns the string unchanged.
    """
    if chr_string.startswith("chr"):
        return chr_string[3:]
    return chr_string


def chr_prefix(chr_string: int | str) -> str:
    """Add the "chr" prefix to a chromosome if not already included.

    If the "chr" prefix is already present, returns the string unchanged.
    """
    if str(chr_string).startswith("chr"):
        return str(chr_string)
    return f"chr{str(chr_string)}"


def to_mega(number: int, dp=1) -> float:
    """Divide by a million to a given number of decimal places.
    
    Useful e.g. to convert base pairs to megabases with dp (1dp by default.)
    """
    return round(number / 1e6, dp)


def resolution_to_int(suffixed_str: str | int) -> int:
    """Convert a string with a "Mb" or "kb" suffix to an int.
    """
    if isinstance(suffixed_str, int):
        return suffixed_str
    if suffixed_str.endswith("Mb"):
        return int(float(suffixed_str[:-2]) * 1e6)
    elif suffixed_str.endswith("kb"):
        return int(float(suffixed_str[:-2]) * 1e3)
    else:
        return int(float(suffixed_str))

def int_to_resolution(resolution: int) -> str:
    """Convert an int resolution into a suffixed with either "kb" or "Mb".

    Rounds to the nearest whole thousand (kb) or million (Mb). 

    """
    if resolution >= 1000000:
        return f"{resolution // 1000000}Mb"
    elif resolution >= 1000:
        return f"{resolution // 1000}kb"
    else:
        return f"{resolution}b"

def get_bin_extent(start: int, end: int, resolution: int) -> tuple[int, int]:
    """Gives the "true" start and end points aligned to resolution bins. 

    The "true" start and end points refer to the start of the bin
    containing the given start argument, and the end of the bin containing
    the given end argument (i.e. the start of the bin after the bin 
    containing the end argument). 

    If "end" is exactly on a bin border, then it considers the true end to 
    be the end of the bin just before the bin containing the end argument. 
    (hence a -1 correction here). 
    """

    trueStart = start // resolution * resolution
    trueEnd = ((end - 1) + resolution) // resolution * resolution 

    return (trueStart, trueEnd)
