import numpy as np
import fmo


def test_hamiltonian_symmetric_published_values():
    H = fmo.hamiltonian()
    assert np.allclose(H, H.T)
    assert H[0, 1] == -87.7 and H[4, 5] == 81.1 and H[2, 2] == 0.0


def test_exciton_band_width():
    e = fmo.exciton_energies()
    assert 12100 < e.min() < 12250 and 12550 < e.max() < 12750


def test_trace_conserved_without_sink():
    _, p, _ = fmo.simulate(t_ps=0.5, dephasing_cm=50, sink_ps=None)
    assert np.allclose(p.sum(1), 1, atol=1e-6)


def test_dephasing_assisted_transport():
    # transporte asistido por desfase (Mohseni 2008; Plenio y Huelga 2008): con pérdidas, un desfase moderado mejora la eficiencia
    eff = lambda g: fmo.simulate(t_ps=5.0, dt_ps=2e-3, dephasing_cm=g, sink_ps=1.0, loss_ps=1000.0)[2][-1]
    assert eff(100.0) > eff(0.0)
