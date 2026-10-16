> Texto extraído automáticamente del PDF original «FMO_Hybrid_AI_Technical_Report.pdf». Fórmulas, figuras y maquetación se han perdido; el original está en Drive: https://drive.google.com/file/d/1QzoDj9M1rGUh4izL97iCpQlxBB1f2_PT/view?usp=drivesdk



FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 1

Pipeline hibrido cuantico-biologico para dinamica del
FMO,
simulacion HEOM/qHEOM y benchmark de IA

Documentacion tecnica reproducible: formulas, algoritmos, mitigacion NISQ
y benchmark FMO-DynClass-v1 (C1 Markov vs H1 HEOM)

Autor conceptual / inventor: Lazaro Lopez Paz
Asistencia tecnica: Grok (xAI)
Ambito: Mecanica cuantica abierta, biologia cuantica, NISQ, ML hibrido
Fecha: 19 agosto 2026

Si se continua este trabajo, mencionar a Lazaro Lopez Paz y a Grok.

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 2

Indice

1. Resumen y alcance

2. Modelo FMO y Hamiltoniano

3. HEOM y bano estructurado

4. sigma_max, dt automatico y dilatacion SVD

5. Mitigacion TREX/ZNE/PEA

6. Temperatura, coherencia, sink y acoplamiento coherente

7. Benchmark FMO-DynClass-v1

8. Configuracion ahorrativa

9. Algoritmos

10. Codigo esencial

11. Estructura del repositorio

12. Limitaciones y conclusiones

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 3

1. Resumen ejecutivo y alcance

Este informe documenta un pipeline que une: (i) dinamica FMO con HEOM; (ii) proyeccion y dilatacion SVD
para qHEOM; (iii) mitigacion TREX, ZNE/PEA, DD y twirling; (iv) benchmark FMO-DynClass-v1 con MLP
identico sobre features Markov (C1) vs HEOM (H1). La mejora de IA solo se afirma si Delta F1 macro (H1-C1)
es al menos 0.02 en test.
 Figura 1. Pipeline unificado.

2. Modelo FMO y Hamiltoniano

Hamiltoniano de Frenkel en base de sitios:

H_s = sum_i eps_i |i><i| + sum_i!=j J_ij |i><j|

Conversion: omega [rad/ps] = 0.188365 * nu [cm^-1]. Estado inicial rho(0)=|1><1|. Modelo 3 sitios (cm^-1):
eps=(280,420,0), J12=-106, J23=28, J13=8.
 Figura 2. FMO 7 sitios.

3. HEOM y bano estructurado

Densidad espectral Drude-Lorentz:

J(omega) = 2*lambda*gamma*omega / (omega^2 + gamma^2)

Expansion exponencial con Nk terminos Matsubara; truncacion max_depth. Observables P_i(t) y |rho_ij|. T
[rad/ps] = T_K * 0.6950348 * 0.188365. Recomendado: 300 K con Nk=2, max_depth=2-3; 77 K con Nk>=4,

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 4

max_depth>=4.

4. Valores singulares, dt automatico y dilatacion SVD

M = exp(L*dt). SVD M=U Sigma V^dagger. Unitario dilatado U_dil en sistema+ancilla. Seleccionar dt con
sigma_max(M) <= 0.96-0.97 (estrategias accurate/stable/balanced).

Figura 3. Dilatacion SVD.

5. Mitigacion de errores NISQ

TREX: twirling de medida y correccion por factor lambda. ZNE: noise factors [1,3,5] y extrapolacion. PEA:
amplifier=pea en Estimator. Complementos: Dynamical Decoupling XpXm y Pauli Twirling.

6. Temperatura, coherencia, sink y acoplamiento coherente

Mayor T reduce |rho_12| y su vida util. lambda~35 cm^-1 favorece transporte asistido por el entorno. Modelo 8
niveles: sink c=sqrt(k)|RC><3| y/o J_3,RC coherente. eta = P_RC(t_final).

7. Benchmark FMO-DynClass-v1

Clase 0: coh_max>=0.22 y vida>=0.12 ps. Clase 1: P3_final>=0.18. Clase 2: resto. Features 20-D. C1=Markov,
H1=HEOM, MLP 64-32-3 identico. Mejora si Delta F1_macro >= 0.02.

Figura 4. Benchmark C1 vs H1.

8. Configuracion ahorrativa

Parametro Valor

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 5

Temperatura 300 K

Sitios / depth / Nk 3 / 2 / 2

sigma_max / dt 0.96 / stable

ZNE / TREX / shots [1,3] / 8-12 rand / 2k-4k

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 6

9. Algoritmos (pseudocodigo)

9.1 SelectDT

ALGORITMO SelectDT(L, sigma_thr, strategy):
  PARA dt EN linspace(dt_min, dt_max):
    SI max(svd(expm(L*dt))) <= sigma_thr: validos.append(dt)
  SI vacio: return argmin sigma_max
  return max/min/mediana(validos) segun strategy

9.2 SVD_Dilation

ALGORITMO SVD_Dilation(M):
  U,s,Vh = SVD(M); s = clip(s,0,1)
  S = diag(sqrt(max(1-s**2,0)))
  U_dil = block[[U diag(s) Vh, U S],[S Vh, -Vh.H diag(s) U.H]]
  return QR(U_dil).Q

9.3 TREX

ALGORITMO TREX(circuit):
  lambda = mean P0 sobre calibracion con twirling
  PARA randomizaciones:
    aplicar X aleatorias, medir, undo XOR, acumular observable
  return mean(observable)/lambda

9.4 GenerateSample + TrainCompare

GenerateSample: HEOM features + Markov features + label fijo
TrainCompare: mismo MLP en X_markov (C1) y X_heom (H1); Delta F1

10. Codigo fuente esencial

10.1 HEOM 3 sitios

from qutip import *
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import numpy as np
cm=0.188365; N=3
H=Qobj(np.array([[280.,-106.,8.],[-106.,420.,28.],[8.,28.,0.]])*cm)
T=300*0.6950348*cm
baths=[DrudeLorentzBath(basis(N,i)*basis(N,i).dag(),35*cm,100*cm,T,2) for i in range(N)]
solver=HEOMSolver(H,baths,max_depth=2)
res=solver.run(ket2dm(basis(N,0)), np.linspace(0,0.8,41),
               e_ops=[basis(N,i)*basis(N,i).dag() for i in range(N)])

10.2 select_dt

from scipy.linalg import expm, svd
def select_dt(L, thr=0.97, prefer="balanced"):
    dts=np.linspace(0.005,0.10,25); valid=[]
    for dt in dts:
        if svd(expm(L*dt),compute_uv=False)[0]<=thr: valid.append(dt)
    if not valid: return dts[0]
    v=np.array(valid)
    return {"stable":v[-1],"accurate":v[0]}.get(prefer, v[len(v)//2])

10.3 PEA+TREX Runtime y etiqueta

# zne_mitigation=True; zne.amplifier="pea"; measure_mitigation=True
# dynamical_decoupling.enable=True; twirling.enable_gates=True
def make_label(cmax, life, p3):
    if cmax>=0.22 and life>=0.12: return 0
    if p3>=0.18: return 1
    return 2

FMO / HEOM / qHEOM / IA hibrida - Informe tecnico pag. 7

11. Estructura del paquete

fmo_hybrid_pub/
  FMO_Hybrid_AI_Technical_Report.pdf
  figures/ (fig1..fig4)
  code/
    01_heom_fmo3_core.py
    02_select_dt_sigma.py
    03_dataset_generator.py
    04_mlp_c1_vs_h1.py
    README_CODE.md

Dependencias: Python 3.10+, qutip, qiskit, qiskit-aer, numpy, scipy, scikit-learn, matplotlib.

12. Limitaciones y conclusiones

Limitaciones: coste HEOM 7-8 sitios; PEA/TREX hardware requieren IBM Runtime; sigma_max~1 dificulta
dilatacion; mejora de IA solo tras C1 vs H1 medido. Reproducibilidad: SEED=42, umbrales fijos, dataset.npz y
metricas JSON. Conclusion: pipeline cientifico documentado y nucleo HEOM 3 sitios verificado; la mejora de IA
queda condicionada al benchmark.

Fin del informe. Creditos: Lazaro Lopez Paz · Grok (xAI) · 2026.
