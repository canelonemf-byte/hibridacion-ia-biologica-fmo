"""Eficiencia de transferencia al sumidero (sitio 3) frente a la tasa de desfase."""
import json
import fmo
res = {g: float(fmo.simulate(t_ps=5.0, dt_ps=2e-3, dephasing_cm=g, sink_ps=1.0, loss_ps=1000.0)[2][-1]) for g in (0, 10, 50, 100, 300, 1000)}
print(json.dumps({"modelo": "Adolphs-Renger 2006 + Haken-Strobl (referencia pública)", "eficiencia_5ps_por_desfase_cm": res, "energias_excitones_cm": [round(x, 1) for x in fmo.exciton_energies()]}, indent=2))
