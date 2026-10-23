"""fmo: dinámica de excitones en el complejo FMO (7 sitios) con un modelo de Lindblad de desfase puro.

Hamiltoniano (cm^-1): Adolphs J., Renger T. (2006) Biophys. J. 91, 2778, doi:10.1529/biophysj.105.079483,
en la forma reproducida por Ishizaki A., Fleming G.R. (2009) PNAS 106, 17255, doi:10.1073/pnas.0908989106
(energías de sitio absolutas 12410, 12530, 12210, 12320, 12480, 12630, 12440 cm^-1).
Modelo de dinámica (Haken-Strobl con sumidero en el sitio 3): Mohseni et al. (2008) J. Chem. Phys. 129, 174106,
doi:10.1063/1.3002335; Plenio y Huelga (2008) New J. Phys. 10, 113019, doi:10.1088/1367-2630/10/11/113019.
Es un modelo de referencia público, no los datos del autor ni un cálculo HEOM.

Autor del código: Lázaro López Paz, El Prat de Llobregat, Barcelona, España, con asistencia de IA.
ORCID 0000-0001-8601-1963. Licencia MIT.
"""
import numpy as np

SITE_ENERGIES = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440], float)
_UPPER = {(0, 1): -87.7, (0, 2): 5.5, (0, 3): -5.9, (0, 4): 6.7, (0, 5): -13.7, (0, 6): -9.9,
          (1, 2): 30.8, (1, 3): 8.2, (1, 4): 0.7, (1, 5): 11.8, (1, 6): 4.3,
          (2, 3): -53.5, (2, 4): -2.2, (2, 5): -9.6, (2, 6): 6.0,
          (3, 4): -70.7, (3, 5): -17.0, (3, 6): -63.3,
          (4, 5): 81.1, (4, 6): -1.3, (5, 6): 39.7}
CM_TO_RAD_PS = 2 * np.pi * 2.99792458e-2  # 1 cm^-1 -> rad/ps


def hamiltonian(offset=12210.0):
    H = np.diag(SITE_ENERGIES - offset)
    for (i, j), v in _UPPER.items():
        H[i, j] = H[j, i] = v
    return H


def exciton_energies():
    return np.linalg.eigvalsh(hamiltonian(0.0))


def simulate(t_ps=5.0, dt_ps=1e-3, dephasing_cm=0.0, sink_site=2, sink_ps=1.0, loss_ps=None, start_site=0):
    """Integra d rho/dt = -i[H, rho] + desfase + sumidero (RK4). Devuelve (t, poblaciones, población_sumidero).
    dephasing_cm: tasa de desfase gamma en cm^-1; sink_ps: tiempo de captura en el sitio 3 (índice 2)."""
    H = hamiltonian() * CM_TO_RAD_PS
    g = dephasing_cm * CM_TO_RAD_PS
    k = 1.0 / sink_ps if sink_ps else 0.0
    kl = 1.0 / loss_ps if loss_ps else 0.0
    n = 7
    rho = np.zeros((n, n), complex); rho[start_site, start_site] = 1
    P = np.zeros(n); P[sink_site] = 1

    def f(r):
        d = -1j * (H @ r - r @ H)
        off = r - np.diag(np.diag(r))
        d -= g * off                       # desfase puro (Haken-Strobl)
        d -= 0.5 * k * (np.outer(P, np.ones(n)) * r + r * np.outer(np.ones(n), P))
        d -= kl * r                        # pérdida (recombinación) uniforme opcional
        return d

    steps = int(round(t_ps / dt_ps)); ts = np.arange(steps + 1) * dt_ps
    pops = np.empty((steps + 1, n)); sink = np.empty(steps + 1); s = 0.0
    pops[0] = np.real(np.diag(rho)); sink[0] = 0
    for i in range(steps):
        k1 = f(rho); k2 = f(rho + dt_ps / 2 * k1); k3 = f(rho + dt_ps / 2 * k2); k4 = f(rho + dt_ps * k3)
        pop_s_before = np.real(rho[sink_site, sink_site])
        rho = rho + dt_ps / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s += k * dt_ps * 0.5 * (pop_s_before + np.real(rho[sink_site, sink_site]))
        pops[i + 1] = np.real(np.diag(rho)); sink[i + 1] = s
    return ts, pops, sink
