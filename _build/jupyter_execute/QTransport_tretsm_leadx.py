#!/usr/bin/env python
# coding: utf-8

# ## Transporte en Semimetal Topoógico del Nudo Trebol
# 
# En el modelo tenemos un hamiltoniano effectivo de la aproximación de amarre fuerte, que se produce con los primos relativos (3,2), donde se incluye simetría $\mathcal{PT}$ y simetría de electrón-hueco $\mathcal{\Xi}$. Primero se plantea el hamiltoniano continuo a bajas energias, el cual esta asociado con las funciones:
# 
# \begin{align}
# a_1(\vec{k}) &=k_x^3-3k_Xk_y^2+k_z^2-(m-0.5k^2)^2 \quad ; \quad a_3(\vec{k})= 3k_x^2k_y-k_y^3+k_z(2m-k^2) \\
# H(\vec{k}) &= a_1\cdot \sigma_x + a_3\cdot \sigma_z +m_z\cdot \sigma_z 
# \end{align}
# En este sistema finito analizamos la conductancia y las estructura de bandas en una terminal con el eje periódico en una dirección. 
# Las unidades de energía en este sistema $\hbar v_f = 1$
# 
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.201305

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import kwant
import kwant.continuum

import ipywidgets
from tqdm.notebook import tqdm

pi = np.pi


# In[ ]:


a = 1.0
#Hamiltoniano continuo
Weyl_ham3D_trebol = """
    (k_x*k_x*k_x-3*k_x*k_y*k_y+k_z*k_z-(m-0.5*k_x*k_x-0.5*k_y*k_y-0.5*k_z*k_z)*(m-0.5*k_x*k_x-0.5*k_y*k_y-0.5*k_z*k_z) )*sigma_x 
    + (3*k_x*k_x*k_y-k_y*k_y*k_y +k_z*(2*m-k_x*k_x-k_y*k_y-k_z*k_z)+mz )*sigma_z
    + V(x,y,z)*sigma_0
"""

wire_template = kwant.continuum.discretize(Weyl_ham3D_trebol,grid=a)


# In[ ]:


# Definir bloque periodico en una direccion, y de longitud L, ancho W y altura Z 
def kwantsyst_leadx(W=20,L=20,Z =20):
    def Shape(site):
        (x, y, z) = site.pos
        return (0 <= y < W and 0 <= x < L and 0 <= z < Z)

    # Definir bloque peri\'odico en y
    def lead_shape_y(site):
        (x, y, z) = site.pos
        return (0 <= y < W and 0 <= z < Z )

    syst = kwant.Builder()
    syst.fill(wire_template, Shape, (0, 0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0, 0]))
    lead.fill(wire_template, lead_shape_y, (0, 0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst

def kwantsyst_leady(W=20,L=20,Z =20):
    def Shape(site):
        (x, y, z) = site.pos
        return (0 <= y < W and 0 <= x < L and 0 <= z < Z)

    # Definir bloque peri\'odico en y
    def lead_shape_y(site):
        (x, y, z) = site.pos
        return (0 <= x < L and 0 <= z < W )

    syst = kwant.Builder()
    syst.fill(wire_template, Shape, (0, 0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([0,-a, 0]))
    lead.fill(wire_template, lead_shape_y, (0, 0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst


def kwantsyst_leadz(W=20,L=20,Z =20):
    def Shape(site):
        (x, y, z) = site.pos
        return (0 <= x < L and 0 <= y < W and 0 <= z < Z)

    # Definir bloque peri\'odico en y
    def lead_shape_y(site):
        (x, y, z) = site.pos
        return (0 <= x < L and 0 <= y < W )

    syst = kwant.Builder()
    syst.fill(wire_template, Shape, (0, 0, 0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([0, 0, -a]))
    lead.fill(wire_template, lead_shape_y, (0, 0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    return syst


# In[ ]:


syst = kwantsyst_leadx()


# In[ ]:


#visualizar nuestro bloque finito de syst
fig = plt.figure()
ax = kwant.plot(syst)
ax.savefig("trenl_leadx.png")


# In[ ]:


# Potenciales, parametros del modelo y calculo de Conductividad
def potential(x,y,z):
    return 0*x+0*y+0*z

DATA = {"m":2.8,
           "mz":0,
       "V":potential}
#Calculo de la conductancia
def plot_conductance(E):
    global syst
    global DATA_25
    smatrix = kwant.smatrix(syst, E, params=DATA_25)
    T = smatrix.transmission(1, 0) 
    return E,T


# In[ ]:


k_node = [-pi, -pi/2, 0, pi/2, pi]
klabel = [ r"$-\pi$",r"$-\frac{\pi}{2}$",0,r"$\frac{\pi}{2}$",r"$\pi$"]


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Estructura de banda en un lead\nplt.figure()\nax = kwant.plotter.bands(syst.leads[0], params=DATA,\n                    momenta=np.linspace(-pi, pi, 201),\n                    show=False)\nplt.ylim(-1,1)\nplt.xlabel("$k_x$")\nplt.xticks(k_node,klabel)\nax.savefig("2502bs_lnr_tsm_leadx_20x20.png")')

