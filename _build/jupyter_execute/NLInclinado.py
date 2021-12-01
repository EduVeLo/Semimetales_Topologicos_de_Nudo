#!/usr/bin/env python
# coding: utf-8

# ## Ahora estudiamos el código asociado al Nodal Line inclinado
# Hamiltoniano continuo de bulto para un semimetal topológico de línea inclinaod
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
from pylab import *
from scipy.optimize import fsolve
from multiprocessing import Pool
# Cargar librería de pythtb
from pythtb import *
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


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


# In[3]:


# Definimos las funciones donde queremos calcular los nodos
def f(k,k_x,m0=2.8, mz=0):
    suma =  cos(k_x) + cos(k[0]) + cos(k[1]) - m0 +mz
    return [sin(k_x) + sin(k[1]),sin(k[0]) + suma]
# vemos que la funcion f[0] es distinto de a1 porque al querer calcular los nodos es la misma funcio\'n


# In[4]:


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


# In[5]:


get_ipython().run_cell_magic('time', '', '# Exploramos las energías de Fermi del bulto\n# Se varía el parametro m_0 asociado al mapeo\nNN = 301\nmesh1= []\nm0 = [1.2,2,2.75]\nx = linspace(-pi,pi, NN)\nfor i in m0:\n    mesh1.append(Nodos(x ,i,0))\n    \nmesh1tb = array(mesh1)\nmesh1tb = mesh1tb / (2*pi)\n\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle(\'Transiciones en el modelo de línea nodal inclinada con $m_z=0$\', fontsize = 18)\n#---- First subplot\nl = len(m0)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection=\'3d\')\n    ax.scatter( mesh1tb[i].T[0], mesh1tb[i].T[1], mesh1tb[i].T[2])\n    ax.set_title(f\'$m_0$={m0[i]}\')\n    ax.set_xlabel(\'$k_x$\', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_ylabel(\'$k_y$\', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_zlabel(\'$k_z$\', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\nfig.savefig("TransicionesTilNLsinnMasa_1.pdf")')


# In[6]:


get_ipython().run_cell_magic('time', '', "# Estados de Fermi en el bulk para m_0=1.4\n# Se varía el termino de masa m_z\nNN = 300\nmesh2= []\nmz = [0.4, 0.2, 0]\nx = linspace(-pi,pi, NN)\nfor i in mz:\n    mesh2.append(Nodos(x ,1.4,i))\nmesh2tb = array(mesh2)\nmesh2tb = mesh2tb / (2*pi)\n\nfor i in range(len(mesh2tb)):\n    fig = plt.figure()\n    ax = Axes3D(fig)\n    ax.scatter( mesh2tb[i].T[0] ,mesh2tb[i].T[1],mesh2tb[i].T[2] )\n\n    ax.set_xlabel('kx')\n    ax.set_ylabel('ky')\n    ax.set_zlabel('kz')")


# In[7]:


get_ipython().run_cell_magic('time', '', "# Estados de Fermi en el bulk para m_0=2.8\n# Se varía el termino de masa m_z\nNN = 300\nmesh3= []\nmz = [0, -0.5, -0.7]\nx = linspace(-pi,pi, NN)\nfor i in range(len(mz)):\n    mesh3.append(Nodos(x ,2.8,mz[i]))\nmesh3tb = array(mesh3)\nmesh3tb = mesh3tb / (2*pi)\n\nfor i in range(len(mesh3tb)):\n    fig = plt.figure()\n    ax = Axes3D(fig)\n    ax.scatter( mesh3tb[i].T[0] ,mesh3tb[i].T[1],mesh3tb[i].T[2] )\n\n    ax.set_xlabel('kx')\n    ax.set_ylabel('ky')\n    ax.set_zlabel('kz')")


# In[8]:


get_ipython().run_cell_magic('time', '', 'proc=Pool()\nlabel  = [ r"$-X$", r"$\\Gamma$", r"$X$"]\nlabel2 = [ r"$-Y$", r"$\\Gamma$", r"$Y$"]\nm0=[2.5]\nfor i in m0:\n    ############### Definimos nuestro tight binding model #############\n    syst = make_tbsys(1, i, 0)\n    ############### Iniciamos un subplot ##############################\n    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))\n    fig.tight_layout(pad=2.0)\n    fig.suptitle(f\'Espectro de bandas para $m_0={i}$ y $m_z=0$\', fontsize=14)\n    \n    for j in range(3):\n        # sistema finito en la direccion j\n        # 200 slabs y condiciones de frontera no periodica\n        cut_j_syst = syst.cut_piece( 200, j, glue_edgs=False)\n        path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\n        k_vec, k_dist, k_node = cut_j_syst.k_path(path,401,report=False)\n        Evls = proc.map(cut_j_syst.solve_one,k_vec)\n        Evls = array(Evls)\n        Evls = Evls.T\n        #Es recomendable guardar estos valores\n        file = open(f\'ev_m{i}_mz0_edge{j}\',"w")\n        for En in Evls:\n            np.savetxt(file, En)\n        file.close()\n        ################# Graficar el espectro de bandas ################## \n        for n in range( len(Evls)):\n            ax[j].plot(k_dist,Evls[n],\'-k\',alpha=0.2)\n\n        # Colocamos una etiqueta al eje y\n        ax[j].set_ylabel("Energía")\n        # Colocamos los xticks en los puntos del path\n        ax[j].set_xticks(k_node)\n        # Dibujamos líneas verticales en cada xtick\n        for n in range(len(k_node)):\n            ax[j].axvline(x=k_node[n], lw=0.5, color=\'k\')\n        # Especificamos los límites de graficación en el eje de las abcisas\n        ax[j].set_ylim(-3,3)\n        if (j == 0):\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_y$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label2)\n        else :\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_x$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label)\n           \n    # Ajustamos los ejes y etiquetas antes de guardar la figura\n    fig.tight_layout()\n    # Guardamos la figura como un pdf\n    fig.savefig(f\'BSTilNLm{i}_mz0_edge{j}.pdf\')')


# In[ ]:




