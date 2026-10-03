# Vendored from sircuitenum/utils.py (GPL-3.0): run_with_timeout only.
"""Timeout helper used by the vendored solver modules."""

from func_timeout import FunctionTimedOut, func_timeout


def run_with_timeout(func, args=(), kwargs=None, timeout=1):
    """
    Runs the specified function with a timeout

    Args:
        func (function): function to run
        args (tuple, optional): arguments to the function. Defaults to ().
        kwargs (_type_, optional): kwarguments to the function. Defaults to None.
        timeout (int, optional): timeout in minutes. Defaults to 1.
    Raises:
        KI: KeyboardInterrupt if interrupted by user
    Returns:
        None if didn't return, else the function output
    """
    if kwargs is None:
        kwargs = {}

    # Pass a zero-argument closure to func_timeout.  On timeout the library
    # formats the supplied args/kwargs into its exception message; symbolic
    # matrices can take minutes to stringify and can raise again inside the
    # timeout thread.  Keeping those objects in the closure makes timeout
    # reporting bounded without changing the function call.
    def call_with_arguments():
        return func(*args, **kwargs)

    try:
        return func_timeout(60*timeout, call_with_arguments)
    except FunctionTimedOut:
        return None
    except KeyboardInterrupt as KI:
        raise KI
