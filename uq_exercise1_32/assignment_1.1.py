import numpy as np
import numpy.typing as npt

PATH_TO_GRADES_FILE = "data/G.txt"

def load_grades(filename: str) -> npt.NDArray:
    # TODO: read grades from the file.
    # ====================================================================
    grades = np.loadtxt(filename, dtype=float)
    # ====================================================================
    return grades


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    n = len(array)
    mean = (1 / n) * sum(array)
    # Dividing by (n - 1) is equivalent to ddof = 1
    var = (1 / (n - 1)) * sum((x - mean) ** 2 for x in array)
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean = np.mean(array)
    # Source: https://numpy.org/devdocs/reference/generated/numpy.var.html
    # "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
    # where N represents the number of elements. By default ddof is zero.
    var = np.var(array, ddof=ddof)
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    grades = load_grades(filename=PATH_TO_GRADES_FILE)
    print(f"grades {grades}")
    mean_python, var_python = python_compute(array=grades)
    print(f"mean_python {mean_python}, var_python {var_python}")
    mean_numpy_ddof_0, var_numpy_ddof_0 = numpy_compute(array=grades, ddof=0)
    print(f"mean_numpy_ddof_0 {mean_numpy_ddof_0}, var_numpy_ddof_0 {var_numpy_ddof_0}")
    mean_numpy_ddof_1, var_numpy_ddof_1 = numpy_compute(array=grades, ddof=1)
    print(f"mean_numpy_ddof_1 {mean_numpy_ddof_1}, var_numpy_ddof_1 {var_numpy_ddof_1}")
    # ====================================================================
