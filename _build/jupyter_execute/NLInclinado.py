#!/usr/bin/env python
# coding: utf-8

# ## Semimetal Topologico de Nodal Line inclinado 
# Hamiltoniano continuo de bulto para un semimetal topológico de línea nodal inclinada, este modelo corresponde a tomar los valores p=q=1.
# 
# \begin{align}
# H(\vec{k}) =& a_1(\vec{k}) \sigma_x + a_3(\vec{k})\sigma_z\\
# H(\vec{k}) =&(k_x+k_z)\sigma_x +(k_y +  m-\frac{k^2}{2})\sigma_z+m_z\sigma_z
# \end{align}
# Como se había definido anteriormente a nuestro modelo general de hamiltonianos continuo resulto del caso de p=1 y q=1. Ahora proponemos nuestras funciones para el hamiltoniano de red:
# \begin{align}
# a_1(\vec{k}) &= \sin k_x + \sin k_z \\
# a_3(\vec{k}) &= \sin k_y + \cos k_x + \cos k_y + \cos k_z - m_0
# \end{align}
# 
# Realizamos el cambio de $\sin$ y $\cos$ por exponenciales
# \begin{align}
# H(\vec{k}) =\left[\begin{array}{cc}
# \frac{1}{2}\bigg(-i(e^{ik_y} - e^{-ik_y}) + e^{ik_x}+e^{-ik_x}+e^{ik_y}+e^{-ik_y}+e^{ik_z}+e^{-ik_z} \bigg) -m_0 & 
# \frac{1}{2i}\bigg( e^{ik_x} - e^{-ik_x} + e^{ik_z} - e^{-ik_z}\bigg)\\
#  \frac{1}{2i}\bigg(e^{ik_x} - e^{-ik_x} + e^{ik_z} - e^{-ik_z} \bigg) & 
# \frac{-1}{2}\bigg(-i(e^{ik_y} - e^{-ik_y}) + e^{ik_x}+e^{-ik_x}+e^{ik_y}+e^{-ik_y}+e^{ik_z}+e^{-ik_z} \bigg) + m_0
# \end{array}\right]
# \end{align}

# In[1]:


# Cargas librerías de python numérico: numpy, scipy, matplotlib
# fsolve encuentra los nodos para cualquier funcion
from pylab import *
from scipy.optimize import fsolve
from multiprocessing import Pool
# Cargar librería de pythtb
from pythtb import *
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


# Definimos las funciones donde queremos calcular los nodos
def f(k,k_x,m0=2.8, mz=0):
    suma =  cos(k_x) + cos(k[0]) + cos(k[1]) - m0 +mz
    return [sin(k_x) + sin(k[1]),sin(k[0]) + suma]
# vemos que la funcion f[0] es distinto de a1 porque al querer calcular los nodos es la misma funcio\'n


# In[3]:


# creamos una funcio\'n que genere una malla de nodos(E=0, Fermi Energy)
def Nodos(k_x, m0=2.8,mz=0):
    k0 = [[-3, 3],[3, -3]]
    k0.append([-1,1])
    k0.append([1,-1])
    k0.append([-4,4])
    k0.append([4,-4])
    k0.append([-0.1,0.1])
    k0.append([0.1,-0.1])
    k0.append([-0.05,0.05])
    k0.append([0.05,-0.05])
    k0.append([-0.0025,0.0025])
    k0.append([0.0025,-0.0025])    
    mesh = []
    
    for j in range(len(k_x)):
        K = []
        for i in range(len(k0)):
            root = fsolve(f,k0[i],args=(k_x[j],m0,mz))
            flag1 = isclose(f(root,k_x[j],m0,mz),[0,0])
            if flag1[0] == True and flag1[1] == True:
                while abs(root[0]) > pi:
                        if root[0] > 0:
                            root[0]-=2*pi
                        else:
                            root[0]+=2*pi
                while abs(root[1]) > pi:
                        if root[1] > 0:
                            root[1]-=2*pi
                        else:
                            root[1]+=2*pi
                k_y, k_z = root
                k = [k_x[j], k_y, k_z]
                K.append(k)
            
        # Eliminamos las raíces iguales
        for m in range(len(K)):
            l=0
            for n in range(len(K)):
                if m != n and n > m:
                    flag = isclose(K[m],K[n])
                    if flag[1] == True and flag[2] == True:
                        l+=1
            if l == 0:
                mesh.append(K[m])
            
    return array(mesh)


# ## Estados de Energía Cero

# In[4]:


get_ipython().run_cell_magic('time', '', '# Exploramos las energías de Fermi del bulto\n# Se varía el parametro m_0 asociado al mapeo\nNN = 201\nm0 = [ 1, 1.2, 2.8, 3.2]\nmesh = []\nl=0\nfor i in range( len(m0)):\n    mesh1=Nodos( NN, m0[i], 0)\n    if ( len( mesh1) != 0):\n        mesh1tb = mesh1 / (2*pi)\n        mesh.append(mesh1tb)\n        l+=1\n    else :\n        print(f\'No hay Estados con E=0 para $m_0$={m0[i]}\')\n# Generamos un subplot de los nodos en el espacio de momentos\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle(\'Transiciones en el modelo de Línea Nodal inclinada\', fontsize = 18)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection=\'3d\')\n    ax.scatter( mesh[i].T[0], mesh[i].T[1], mesh[i].T[2])\n    ax.set_title(f\'$m_0$={m0[i]}\')\n    ax.set_xlabel(\'$k_x$\', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_ylabel(\'$k_y$\', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_zlabel(\'$k_z$\', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.view_init(20,-45)\nfig.tight_layout()\nfig.savefig("TransicionesTilNLsinnMasa_1.pdf")')


# In[5]:


#Proyecciones de las superficies de Fermi en los planos canonicos 
fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
X=['x', 'y', 'z']
l = len(X)
k=1
for m0 in range( len(mesh)):
    for i in range( l):
        for j in range( l):
            if (j > i ):
                ax = fig.add_subplot(1, l, k)
                ax.scatter( mesh[m0].T[i],mesh[m0].T[j])
                ax.set_xlabel(f'$k_{X[i]}$', fontsize = 14)
                ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
                ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
                ax.set_ylabel(f'$k_{X[j]}$', fontsize = 14)
                ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
                ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
                k+=1

    fig.tight_layout()
    fig.savefig(f'FBSNLInc_m{m0}_mz0.png', bbox_inches ='tight')
    plt.show()


# In[6]:


get_ipython().run_cell_magic('time', '', "# Generamos los puntos que nos dan Energía cero en nuestro hamiltoniano continuo\n# Exploración de las energías de Fermi para E=0,con distinto parametro mz\nNN = 101\nmz = [ 0.1, 0.2, 0.5]\nmesh = []\nl=0\nfor i in range(len(mz)):\n    mesh1=Nodos(NN, 2.8,mz[i])\n    if ( len( mesh1) != 0):\n        mesh1tb = mesh1 / (2*pi)\n        mesh.append(mesh1tb)\n        l+=1\n    else :\n        print(f'No hay Estados con E=0 para $m_z$={mz[i]}')\n# Generamos un subplot de los nodos en el espacio de momentos\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle('Transiciones en el modelo de Línea Nodal inclinada $m_0=2.8$', fontsize = 18)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection='3d')\n    ax.scatter( mesh[i].T[0], mesh[i].T[1], mesh[i].T[2])\n    ax.set_title(f'$m_0$={m0[i]}')\n    ax.set_xlabel('$k_x$', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.set_ylabel('$k_y$', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.set_zlabel('$k_z$', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.view_init(20,-45)\nfig.savefig(f'TransicionesNLIncMasa28_mz{i}.pdf')")


# In[7]:


#Proyecciones de las superficies de Fermi en los planos canonicos 
fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
X=['x', 'y', 'z']
l = len(X)
k=1
for mz in range( len(mesh)):
    for i in range( l):
        for j in range( l):
            if (j > i ):
                ax = fig.add_subplot(1, l, k)
                ax.scatter( mesh[m0].T[i],mesh[m0].T[j])
                ax.set_xlabel(f'$k_{X[i]}$', fontsize = 14)
                ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
                ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
                ax.set_ylabel(f'$k_{X[j]}$', fontsize = 14)
                ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
                ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
                k+=1

    fig.savefig(f'FBSNLInc_m{mz}_mz0.png', bbox_inches ='tight')
    plt.show()


# ### Estados de borde en una dirección finita

# In[8]:


# Construir el hamiltoniano de red en 
def make_tbsys(a=1 ,m0=2.8 ,mz=0):
    # Vectores de red
    lat = [[a,0,0],[0,a,0],[0,0,a]]

    # Posición de los sitio de la red 
    # en términos de los vectores de re
    orb = [[0,0,0],[1/2,1/2,1/2]]

    # Se genera un modelo de:
    # 2 dimensiones en el espacio real
    # 2 dimensiones en el espacio recíproco
    # con los vectores de red lat
    # con orbitales en los sitios orb
    TSMK11 = tb_model(3,3,lat,orb)

    # Parámetros del modelo:
    # M,t1,t2
    M =  -m0 + mz
    s =  -0.5J
    c =  0.5

    # Establecer las energías on-site
    TSMK11.set_onsite( [ M, -M])

    # Establecer los hoppings a primeros vecinos
    # (hopping, sitio i, sitio j, [vector de red de la celda donde se encuentra j])
    
    ##términos en la diagonal

    TSMK11.set_hop(c,0,0,[1, 0, 0])
    TSMK11.set_hop(s+c,0,0,[0, 1, 0])
    TSMK11.set_hop(c,0,0,[0, 0, 1])
    TSMK11.set_hop(-c,1,1,[1, 0, 0])
    TSMK11.set_hop(-c-s,1,1,[0, 1, 0])    
    TSMK11.set_hop(-c,1,1,[0, 0, 1])
    # términos fuera de la diagonal
    TSMK11.set_hop(s,1,0,[ 1, 0, 0])
    TSMK11.set_hop(-s,1,0,[ -1, 0,0])
    TSMK11.set_hop(s,1,0,[ 0, 0, 1])
    TSMK11.set_hop(-s,1,0,[ 0, 0,-1])
    return TSMK11


# In[9]:


get_ipython().run_cell_magic('time', '', 'proc=Pool()\nlabel  = [ r"$-X$", r"$\\Gamma$", r"$X$"]\nlabel2 = [ r"$-Y$", r"$\\Gamma$", r"$Y$"]\nm0=[2.5]\nfor i in m0:\n    ############### Definimos nuestro tight binding model #############\n    syst = make_tbsys(1, i, 0)\n    ############### Iniciamos un subplot ##############################\n    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))\n    fig.tight_layout(pad=2.0)\n    fig.suptitle(f\'Espectro de bandas para $m_0={i}$ y $m_z=0$\', fontsize=14)\n    \n    for j in range(3):\n        # sistema finito en la direccion j\n        # 200 slabs y condiciones de frontera no periodica\n        cut_j_syst = syst.cut_piece( 200, j, glue_edgs=False)\n        path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\n        k_vec, k_dist, k_node = cut_j_syst.k_path(path,401,report=False)\n        Evls = proc.map(cut_j_syst.solve_one,k_vec)\n        Evls = array(Evls)\n        Evls = Evls.T\n        #Es recomendable guardar estos valores\n        file = open(f\'ev_m{i}_mz0_edge{j}\',"w")\n        for En in Evls:\n            np.savetxt(file, En)\n        file.close()\n        ################# Graficar el espectro de bandas ################## \n        for n in range( len(Evls)):\n            ax[j].plot(k_dist,Evls[n],\'-k\',alpha=0.2)\n\n        # Colocamos una etiqueta al eje y\n        ax[j].set_ylabel("Energía")\n        # Colocamos los xticks en los puntos del path\n        ax[j].set_xticks(k_node)\n        # Dibujamos líneas verticales en cada xtick\n        for n in range(len(k_node)):\n            ax[j].axvline(x=k_node[n], lw=0.5, color=\'k\')\n        # Especificamos los límites de graficación en el eje de las abcisas\n        ax[j].set_ylim(-3,3)\n        if (j == 0):\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_y$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label2)\n        else :\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_x$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label)\n           \n    # Ajustamos los ejes y etiquetas antes de guardar la figura\n    fig.tight_layout()\n    # Guardamos la figura como un pdf\n    fig.savefig(f\'BSTilNLm{i}_mz0_edge{j}.pdf\')')


# In[ ]:




