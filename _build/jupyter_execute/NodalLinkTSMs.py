#!/usr/bin/env python
# coding: utf-8

# ## Topological Linked Nodal-Line Semimetals
# Consultar un modelo similar con distinta proyeccion en: 
# [Zhongbo Yan et al.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.041103)
# 
# En el modelo tenemos un hamiltoniano effectivo de la aproximación de amarre fuerte, que se produce con los primos relativos (2,2), donde se incluye simetría $\mathcal{PT}$ y simetría de electrón-hueco $\mathcal{\Xi}$. Primero se plantea el hamiltoniano continuo a bajas energias, el cual esta asociado con las funciones:
# \begin{align}
# H(\vec{k}) &= a_1\cdot \sigma_x + a_3\cdot \sigma_z +m_z\cdot \sigma_z\\
# a_1(\vec{k}) =k_x^2-k_y^2+k_z^2&-(m-0.5k^2)^2 \quad ; \quad a_3= 2k_xk_y+k_z(2m-k^2)
# \end{align} 
# En el modelo de red propuesto es:
# \begin{align}
# a_1(\vec{k}) &=\sin^2{k_x}-\sin^2{k_y}+\sin^2{k_z}-(\sum_j\cos{k_j}-m_0)^2 \\
# a_3(\vec{k}) &= 2\sin{k_x}\sin{k_y}+2\sin{k_z}(\sum_j\cos{k_j}-m_0)
# \end{align}
# donde se ha definido $m_0= 3-m$
# Expresamos las exponenciales de senos y cosenos como suma de angulos:
# \begin{align}
# a_1(\vec{k}) =&-\frac{1}{2}\sum_j\cos{2k_j}-\cos{(k_x+k_y)}-\cos{(k_x-k_y)}-\cos{(k_x+k_z)}-\cos{(k_x-k_z)}-
#         \cos{(k_y+k_z)}-\cos{(k_y-k_z)}+2(m_0-1)(\cos{k_x}+\cos{k_z})+2(m_0+1)\cos{k_y}+\frac{1}{2}-m_0^2 \\
# a_3(\vec{k}) =& = \cos (k_x - k_z)-\cos (k_x+k_z) +\sum_j(\sin( k_y + k_j )+\sin (k_y - k_j) ) - 2m_0 \sin k_y \\
# \end{align} 

# In[1]:


from pylab import *
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from pythtb import *
from multiprocessing import Pool
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')


# ## Estados de Energía cero

# In[2]:


# Definimos las funciones donde queremos calcular los nodos
# fenx es la función para hallar el nodo [k_y,k_z] que está asociado al discretizar k_x
def fenx(k,k_x,m0=2.8, mz=0):
    suma =  cos(k_x) + cos(k[0]) + cos(k[1]) - m0
    a1 =  sin(k_x)*sin(k_x) - sin(k[0])*sin(k[0]) + sin(k[1])*sin(k[1]) -suma*suma
    a3 = 2*sin(k_x)*sin(k[0])+2*sin(k[1])*suma + mz
    return [a1 , a3]
#  feny es la función para hallar los nodos [k_x,k_z] que está asociado al discrtizar k_y
def feny(k,k_y,m0=2.8, mz=0):
    suma =  cos(k[0]) + cos(k_y) + cos(k[1]) - m0
    a1 =  sin(k[0])*sin(k[0]) - sin(k_y)*sin(k_y) + sin(k[1])*sin(k[1]) -suma*suma
    a3 = 2*sin(k_y)*sin(k[0])+2*sin(k[1])*suma + mz
    return [a1 , a3]
#  fenz es la función para hallar los nodos [k_x,k_y] que está asociado al discrtizar k_z
def fenz(k,k_z,m0=2.8, mz=0):
    suma =  cos(k_z) + cos(k[0]) + cos(k[1]) - m0
    a1 =  sin(k[0])*sin(k[0]) - sin(k[1])*sin(k[1]) + sin(k_z)*sin(k_z) -suma*suma
    a3 = 2*sin(k[1])*sin(k[0])+2*sin(k_z)*suma + mz
    return [a1 , a3]


# In[3]:


# creamos una funcio\'n que genere una malla de nodos
def Nodos(NN, m0=2.8,mz=0):
    k_j=linspace(-pi, pi, NN)
    k0 = [[-3.2, 3.2],[3.2,-3.2],[-2,2],[2,-2],[-0.0025,0.0025],[0.0025,-0.0025]]  
    mesh = []
    #Calculamos los nodos para x discretizado
    for j in range(len(k_j)):
        K = []
        for i in range(len(k0)):
            root = fsolve(fenx,k0[i],args=(k_j[j],m0,mz))
            flag1 = isclose(fenx(root,k_j[j],m0,mz),[0,0])
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
                k = [k_j[j], k_y, k_z]
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
    #Calculamos los nodos para y discretizado
    for j in range(len(k_j)):
        K = []
        for i in range(len(k0)):
            root = fsolve(feny,k0[i],args=(k_j[j],m0,mz))
            flag1 = isclose(feny(root,k_j[j],m0,mz),[0,0])
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
                k_x, k_z = root
                k = [k_x, k_j[j], k_z]
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
    #calculamos los nodos para z discretizado
    for j in range(len(k_j)):
        K = []
        for i in range(len(k0)):
            root = fsolve(fenz,k0[i],args=(k_j[j],m0,mz))
            flag1 = isclose(fenz(root,k_j[j],m0,mz),[0,0])
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
                k_x, k_y = root
                k = [k_x, k_y, k_j[j]]
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
                
        
    #Eliminanos los nodos iguales
    Mesh = []
    for m in range(len(mesh)):
        l = 0
        for n in range(len(mesh)):
            if m != n and n > m:
                flag = isclose(mesh[m],mesh[n])
                if flag[0] == True and flag[1] == True and flag[2] == True:
                        l+=1
        if l== 0:
            Mesh.append(mesh[m])
    
    return array(Mesh)


# In[4]:


get_ipython().run_cell_magic('time', '', '# Exploramos las energías de Fermi del bulto\n# Se varía el parametro m_0 asociado al mapeo\nNN = 201\nm0 = [ 1, 1.2, 2.8, 3.2]\nmesh = []\nl=0\nfor i in range( len(m0)):\n    mesh1=Nodos( NN, m0[i], 0)\n    if ( len( mesh1) != 0):\n        mesh1tb = mesh1 / (2*pi)\n        mesh.append(mesh1tb)\n        l+=1\n    else :\n        print(f\'No hay Estados con E=0 para $m_0$={m0[i]}\')\n# Generamos un subplot de los nodos en el espacio de momentos\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle(\'Transiciones en el modelo de Nudo de Líneas Nodales enlazadas\', fontsize = 18)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection=\'3d\')\n    ax.scatter( mesh[i].T[0], mesh[i].T[1], mesh[i].T[2])\n    ax.set_title(f\'$m_0$={m0[i]}\')\n    ax.set_xlabel(\'$k_x$\', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_ylabel(\'$k_y$\', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_zlabel(\'$k_z$\', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.view_init(20,-45)\nfig.tight_layout()\nfig.savefig("TransicionesLNLsinMasa_1.pdf")')


# In[ ]:


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
    fig.savefig(f'FBSLNL_m{m0}_mz0.png', bbox_inches ='tight')
    plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Generamos los puntos que nos dan Energía cero en nuestro hamiltoniano continuo\n# Exploración de las energías de Fermi para E=0,con distinto parametro mz\nNN = 101\nmz = [ 0.2, 0.4]\nmesh = []\nl=0\nfor i in range(len(mz)):\n    mesh1=Nodos(NN, 2.8,mz[i])\n    if ( len( mesh1) != 0):\n        mesh1tb = mesh1 / (2*pi)\n        mesh.append(mesh1tb)\n        l+=1\n    else :\n        print(f'No hay Estados con E=0 para $m_z$={mz[i]}')\n# Generamos un subplot de los nodos en el espacio de momentos\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle('Transiciones en el modelo de Linked Nodal line $m_0=2.8$', fontsize = 18)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection='3d')\n    ax.scatter( mesh[i].T[0], mesh[i].T[1], mesh[i].T[2])\n    ax.set_title(f'$m_0$={m0[i]}')\n    ax.set_xlabel('$k_x$', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.set_ylabel('$k_y$', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.set_zlabel('$k_z$', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r'$-\\pi$', r'$-\\pi/2$', r'0', r'$\\pi/2$', r'$\\pi$'])\n    ax.view_init(20,-45)\nfig.savefig(f'TransicionesLNLMasa25_mz{i}.pdf')")


# In[ ]:


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

    fig.savefig(f'FBSLNL_m{mz}_mz0.png', bbox_inches ='tight')
    plt.show()


# ## Estados de borde para una direccion finita

# In[ ]:


# Construimos nuestro systema en pythtb
def make_tbsyst(a=1.0 ,m0=2.8, mz=0):
    # Construccion del hailtoniano
    # Vectores de red
    lat = [[a,0,0],[0,a,0],[0,0,a]]

    # Posición de los sitio de la red 
    # en términos de los vectores de red
    orb = [[0,0,0],[1/2,1/2,1/2]]

    # Se genera un modelo de: Nodal Linked
    # 2 dimensiones en el espacio real
    # 2 dimensiones en el espacio recíproco
    # con los vectores de red lat
    # con orbitales en los sitios orb
    LinkTSM22 = tb_model(3,3,lat,orb)

    # Parámetros del modelo:
    # M, T_2, t_2, m0
    s  = -0.5J
    c  = 0.5
    t_00 = -2*m0*s 
    # Establecer las energías on-site
    LinkTSM22.set_onsite( [ mz, -mz])

    # Establecer los hoppings a primeros vecinos
    # (hopping, sitio i, sitio j, [vector de red de la celda donde se encuentra j])

    # Terminos en la diagonal
    ## sen(k_)
    LinkTSM22.set_hop(2*s,0,0,[ 0, 1, 0]) ## sin k_y 
    LinkTSM22.set_hop(t_00,0,0,[ 0, 0, 1]) ## sin k_z
    ## sen(2k_)
    LinkTSM22.set_hop(0.5*s ,0,0,[ 0, 2, 0])
    LinkTSM22.set_hop(s ,0,0,[ 0, 0, 2])
    ## sen(k_y+...)
    LinkTSM22.set_hop(-1.5*s ,0,0,[ 1, 1, 0]) #k_x + k_y#
    LinkTSM22.set_hop(-1.5*s ,0,0,[ -1, 1, 0])  #k_y - k_x#
    LinkTSM22.set_hop(s ,0,0,[ 0, 1, 1]) #k_y + k_z#
    LinkTSM22.set_hop(s ,0,0,[ 0, -1, 1])   #k_z - k_y#
    LinkTSM22.set_hop(s ,0,0,[ 1, 0, 1]) #k_x + k_z#
    LinkTSM22.set_hop(s ,0,0,[-1, 0, 1])   #k_z - k_x#

    ## sen(k_)
    LinkTSM22.set_hop(-2*s,1,1,[ 0, 1, 0]) ## sin k_y 
    LinkTSM22.set_hop(-t_00,1,1,[ 0, 0, 1]) ## sin k_z
    ## sen(2k_)
    LinkTSM22.set_hop(-0.5*s ,1,1,[ 0, 2, 0])
    LinkTSM22.set_hop(-s ,1,1,[ 0, 0, 2])
    ## sen(k_y+...)
    LinkTSM22.set_hop(1.5*s ,1,1,[ 1, 1, 0]) #k_x + k_y#
    LinkTSM22.set_hop(1.5*s ,1,1,[ -1, 1, 0])  #k_y - k_x#
    LinkTSM22.set_hop(-s ,1,1,[ 0, 1, 1]) #k_y + k_z#
    LinkTSM22.set_hop(-s ,1,1,[ 0, -1, 1])   #k_z - k_y#
    LinkTSM22.set_hop(-s ,1,1,[ 1, 0, 1]) #k_x + k_z#
    LinkTSM22.set_hop(-s ,1,1,[-1, 0, 1])   #k_z - k_x#

    # Terminos fuera de la diagonal
    ## intracell hopping
    LinkTSM22.set_hop(-1-m0**2 ,0,1,[ 0, 0, 0]) 
    ##
    LinkTSM22.set_hop(m0 - 2*s ,0,1,[ 1, 0, 0])
    LinkTSM22.set_hop(m0 + 2*s ,0,1,[ -1, 0, 0])
    LinkTSM22.set_hop(m0 ,0,1,[ 0, 1, 0])
    LinkTSM22.set_hop(m0 ,0,1,[ 0,-1, 0])
    LinkTSM22.set_hop(m0 ,0,1,[ 0, 0, 1])
    LinkTSM22.set_hop(m0 ,0,1,[ 0, 0,-1]) 
    ##  2k...
    LinkTSM22.set_hop(-0.5*(c+s),0,1,[2 , 0, 0])
    LinkTSM22.set_hop(-0.5*(c-s),0,1,[-2 , 0, 0])
    LinkTSM22.set_hop(-0.5*c ,0,1,[0 , 2, 0])
    LinkTSM22.set_hop(-0.5*c ,0,1,[0 , -2, 0])
    LinkTSM22.set_hop( -c ,0,1,[0 , 0, 2])
    LinkTSM22.set_hop( -c ,0,1,[0 , 0,-2])
    ## cos(k_l+ k_m)
    LinkTSM22.set_hop(-c + 1.5*s,0,1,[ 1, 1, 0])
    LinkTSM22.set_hop(-c - 1.5*s ,0,1,[ -1, -1, 0]) ## cos k_x+k_y
    LinkTSM22.set_hop(-c + 1.5*s,0,1,[ 1, -1, 0])
    LinkTSM22.set_hop(-c- 1.5*s ,0,1,[ -1, 1, 0])  ## cos k_x-k_y
    LinkTSM22.set_hop(-c ,0,1,[ 1, 0, 1])
    LinkTSM22.set_hop(-c ,0,1,[ -1, 0, -1]) ## cos k_x+k_z
    LinkTSM22.set_hop(-c ,0,1,[ 1,0 , -1])
    LinkTSM22.set_hop(-c ,0,1,[-1, 0, 1])  ## cos k_x-k_z
    LinkTSM22.set_hop(-c ,0,1,[ 0, 1, 1])
    LinkTSM22.set_hop(-c ,0,1,[ 0, -1, -1])  ## cos k_y+k_z
    LinkTSM22.set_hop(-c ,0,1,[ 0, -1, 1])
    LinkTSM22.set_hop(-c ,0,1,[ 0, 1, -1])  # cos k_y-k_z
    return LinkTSM22


# In[ ]:


get_ipython().run_cell_magic('time', '', 'proc=Pool()\nlabel  = [ r"$-X$", r"$\\Gamma$", r"$X$"]\nlabel2 = [ r"$-Y$", r"$\\Gamma$", r"$Y$"]\nm0=[1,1.5,2.5,3.2]\nfor i in m0:\n    ############### Definimos nuestro tight binding model #############\n    syst = make_tbsyst(1, i, 0)\n    ############### Iniciamos un subplot ##############################\n    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))\n    fig.tight_layout(pad=2.0)\n    fig.suptitle(f\'Espectro de bandas para $m_0={i}$ y $m_z=0$\', fontsize=14)\n    \n    for j in range(3):\n        # sistema finito en la direccion j\n        # 200 slabs y condiciones de frontera no periodica\n        cut_j_syst = syst.cut_piece( 200, j, glue_edgs=False)\n        path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\n        k_vec, k_dist, k_node = cut_j_syst.k_path(path,401,report=False)\n        Evls = proc.map(cut_j_syst.solve_one,k_vec)\n        Evls = array(Evls)\n        Evls = Evls.T\n        #Es recomendable guardar estos valores\n        file = open(f\'ev_m{i}_mz0_edge{j}\',"w")\n        for En in Evls:\n            np.savetxt(file, En)\n        file.close()\n        ################# Graficar el espectro de bandas ################## \n        for n in range( len(Evls)):\n            ax[j].plot(k_dist,Evls[n],\'-k\',alpha=0.2)\n\n        # Colocamos una etiqueta al eje y\n        ax[j].set_ylabel("Energía")\n        # Colocamos los xticks en los puntos del path\n        ax[j].set_xticks(k_node)\n        # Dibujamos líneas verticales en cada xtick\n        for n in range(len(k_node)):\n            ax[j].axvline(x=k_node[n], lw=0.5, color=\'k\')\n        # Especificamos los límites de graficación en el eje de las abcisas\n        ax[j].set_ylim(-3,3)\n        if (j == 0):\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_y$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label2)\n        else :\n            # Colocamos una etiqueta al eje x \n            ax[j].set_xlabel("Camino en el espacio $k_x$")\n            # Colocamos las etiquetas de los xticks \n            ax[j].set_xticklabels(label)\n           \n    # Ajustamos los ejes y etiquetas antes de guardar la figura\n    fig.tight_layout()\n    # Guardamos la figura como un pdf\n    fig.savefig(f\'BSTilNLm{i}_mz0_edge{j}.pdf\')')

