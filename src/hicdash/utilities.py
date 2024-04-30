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


def suffixed_string_to_int(suffixed_str: str) -> int:
    """ Convert a string with a "Mb" or "kb" suffix to an int.
    """
    if suffixed_str.endswith("Mb"):
        return int(float(suffixed_str[:-2]) * 1e6)
    elif suffixed_str.endswith("kb"):
        return int(float(suffixed_str[:-2]) * 1e3)
    else:
        return int(float(suffixed_str))