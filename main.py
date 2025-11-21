# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)) or len(A.shape) != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0 or A.shape[1] == 0:
        return None
    d = abs(A).diagonal()
    s = abs(A).sum(axis=1) - d
    return np.all(d >= s)


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        return None
    if len(x.shape) != 1 or x.shape[0] != A.shape[1]:
        return None
    if len(b.shape) != 1 or b.shape[0] != A.shape[0]:
        return None
    if sp.sparse.issparse(A):
        r = A @ x - b
    else:
        r = A.dot(x) - b
    return np.linalg.norm(r)