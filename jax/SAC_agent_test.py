import pytest
import SAC_agent as SAC
import numpy as np
from numpy import log, abs, array as arr
import scipy.stats as stats

tol = 1e-7


def test_log_pdf__std_normal():
    rv = stats.norm()
    assert (
        abs(log(rv.pdf(0)) - SAC.log_pdf_std_norm(arr([0]), arr([0]), arr([1]))) < tol
    )
    assert (
        abs(log(rv.pdf(1)) - SAC.log_pdf_std_norm(arr([1]), arr([0]), arr([1]))) < tol
    )
    assert (
        abs(log(rv.pdf(-1.5)) - SAC.log_pdf_std_norm(arr([-1.5]), arr([0]), arr([1])))
        < tol
    )


def test_log_pdf__nonstd_normal():
    rv = stats.norm(loc=5, scale=2)
    assert (
        abs(log(rv.pdf(5)) - SAC.log_pdf_std_norm(arr([5]), arr([5]), arr([2]))) < tol
    )
    assert (
        abs(log(rv.pdf(2)) - SAC.log_pdf_std_norm(arr([2]), arr([5]), arr([2]))) < tol
    )
    assert (
        abs(log(rv.pdf(6.5)) - SAC.log_pdf_std_norm(arr([6.5]), arr([5]), arr([2])))
        < tol
    )


# log(rv.pdf(0)) - SAC.log_pdf_std_norm(arr([0]), arr([0]), arr([1]))
