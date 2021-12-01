#!/usr/bin/env python
# coding: utf-8

# ## Modelo de nudo de trebol de línea Nodal
# Consultar: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.201305 \
# El modelo para el nudo de trebol está asociado con los enteros p=3 y q=2, el polinomio asociado al modelo continuo es:
# \begin{align*} 
# a_1(m,\vec{k})=&k_x^3-3k_xk_y^2+k_z^2-(m-\frac{1}{2}k^2)^2 \\
#  a_3(m,\vec{k})=&-k_y^3+3k_yk_x^2+2k_z(m-\frac{1}{2}k^2)
# \end{align*}
# Entonces para proponer el modelo de red, tomamos:
# \begin{align*}
# 	a_1(m_0,\vec{k})=&\sin{k_x}(1-\cos{k_x})-3\sin{k_x}(1-\cos{k_y})+\sin^2{k_z}-(\sum_j\cos{k_j}-m_0)^2\\
# 	a_3(m_0,\vec{k})=&-\sin{k_y}(1-\cos{k_y})+3\sin{k_y}(1-\cos{k_x})+2\sin{k_z}(\sum_j\cos{k_j}-m_0)
# \end{align*}
# Ahora expresamos el modelo de red en terminos de senos y cosenos en suma de ángulos:
# \begin{align*}
#      a_1=&-2\sin{k_x}-\frac{1}{2}\sin{2k_x}+\frac{3}{2}(\sin{(k_x+k_y)}+\sin{(k_x-k_y)})+\bigg(\sin^2{k_z}-\sum_j\cos^2{k_j}\bigg)\\
#      &-(\cos{(k_x+k_y)}+\cos{(k_x-k_y)}+\cos{(k_x+k_z)}+\cos{(k_x-k_z)}+\cos{(k_z+k_y)}\\
#      &+\cos{(k_z-k_y)})+2m_0\sum_j\cos{k_j}-m_0^2\\
#      =&-2\sin{k_x}-\frac{1}{2}\sin{2k_x}+\frac{3}{2}(\sin{(k_x+k_y)}+\sin{(k_x-k_y)})+\bigg( -1-\frac{(\cos{2k_x}+\cos{2k_y})}{2}-\cos{2k_z} \bigg)\\
#      &-(\cos{(k_x+k_y)}+\cos{(k_x-k_y)}+\cos{(k_x+k_z)}+\cos{(k_x-k_z)}+\cos{(k_z+k_y)}\\
#      &+\cos{(k_z-k_y)})+2m_0\sum_j\cos{k_j}-m_0^2\\
#         a_3=&2\sin{k_y}+\frac{1}{2}\sin{2k_y}-\frac{3}{2}(\sin{(k_x+k_y)}+\sin{(k_y-k_x)})+\sum_j(\sin{(k_z+k_j)}+\sin{(k_z-k_j)}) -2m_0\sin{k_z}
# \end{align*}

# In[1]:


from pylab import *
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


from pythtb import *
from multiprocessing import Pool


# In[3]:


# Definimos las funciones donde queremos calcular los nodos
# Estas funciones están dados por el hamiltoniano de Bloch 
# fenx es la función para hallar el nodo [k_y,k_z] que está asociado al discretizar k_x
def fenx(k,k_x,m0=2.8, mz=0):
    suma =  cos(k_x) + cos(k[0]) + cos(k[1]) - m0
    a1 = -2*sin(k_x)-0.5*sin(2*k_x)+3*sin(k_x)*cos(k[0])+sin(k[1])*sin(k[1])-suma*suma
    a3 = 2*sin(k[0])+0.5*sin(2*k[0])-3*sin(k[0])*cos(k_x)+2*sin(k[1])*suma+mz
    return [a1, a3]
# feny es la función para hallar el nodo [k_x,k_z] que está asociado al discretizar k_y
def feny(k,k_y,m0=2.8, mz=0):
    suma =  cos(k[0]) + cos(k_y) + cos(k[1]) - m0
    a1 = -2*sin(k[0])-0.5*sin(2*k[0])+3*sin(k[0])*cos(k_y)+sin(k[1])*sin(k[1])-suma*suma
    a3 = 2*sin(k_y)+0.5*sin(2*k_y)-3*sin(k_y)*cos(k[0])+2*sin(k[1])*suma+mz
    return [a1, a3]
# fenz es la función para hallar el nodo [k_x,k_y] que está asociado al discretizar k_z
def fenz(k,k_z,m0=2.8, mz=0):
    suma =  cos(k_z) + cos(k[0]) + cos(k[1]) - m0
    a1 = -2*sin(k[0])-0.5*sin(2*k[0])+3*sin(k[0])*cos(k[1])+sin(k_z)*sin(k_z)-suma*suma
    a3 = 2*sin(k[1])+0.5*sin(2*k[1])-3*sin(k[1])*cos(k[0])+2*sin(k_z)*suma+mz
    return [a1, a3]


# In[4]:


# creamos una funcion que genere una malla de nodos
def Nodos(NN, m0=2.8,mz=0):
    k_j=linspace(-pi, pi, NN)
    #Definimos un punto inicial para el calculo de raices
    k0 = [[-3.2, 3.2],[3.2,-3.2],[2,0],[2,2],[-2,2],[2,-2],[-0.0025,0.0025],[0.0025,-0.0025]]  
    #Una variable que guarde los puntos
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


# In[5]:


#Construimos nuestro modelo de amarre fuerte con las expresiones exponenciales
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
    TrNSmK32 = tb_model(3,3,lat,orb)

    # Parámetros del modelo:
    # M, T_2, t_2, m0
    s  = -0.5J
    c  = 0.5
    t_00 = -2*m0*s 
    # Establecer las energías on-site
    TrNSmK32.set_onsite( [ mz, -mz])

    # Establecer los hoppings a primeros vecinos
    # (hopping, sitio i, sitio j, [vector de red de la celda donde se encuentra j])

    # Terminos en la diagonal a_3
    ## sen(k_)
    TrNSmK32.set_hop(2*s,0,0,[ 0, 1, 0]) ## sin k_y 
    TrNSmK32.set_hop(t_00,0,0,[ 0, 0, 1]) ## sin k_z
    ## sen(2k_)
    TrNSmK32.set_hop(0.5*s ,0,0,[ 0, 2, 0])
    TrNSmK32.set_hop(s ,0,0,[ 0, 0, 2])
    ## sen(k_y+...)
    TrNSmK32.set_hop(-1.5*s ,0,0,[ 1, 1, 0]) #k_x + k_y#
    TrNSmK32.set_hop(-1.5*s ,0,0,[ -1, 1, 0])  #k_y - k_x#
    TrNSmK32.set_hop(s ,0,0,[ 0, 1, 1]) #k_y + k_z#
    TrNSmK32.set_hop(s ,0,0,[ 0, -1, 1])   #k_z - k_y#
    TrNSmK32.set_hop(s ,0,0,[ 1, 0, 1]) #k_x + k_z#
    TrNSmK32.set_hop(s ,0,0,[-1, 0, 1])   #k_z - k_x#

    ## sen(k_)
    TrNSmK32.set_hop(-2*s,1,1,[ 0, 1, 0]) ## sin k_y 
    TrNSmK32.set_hop(-t_00,1,1,[ 0, 0, 1]) ## sin k_z
    ## sen(2k_)
    TrNSmK32.set_hop(-0.5*s ,1,1,[ 0, 2, 0])
    TrNSmK32.set_hop(-s ,1,1,[ 0, 0, 2])
    ## sen(k_y+...)
    TrNSmK32.set_hop(1.5*s ,1,1,[ 1, 1, 0]) #k_x + k_y#
    TrNSmK32.set_hop(1.5*s ,1,1,[ -1, 1, 0])  #k_y - k_x#
    TrNSmK32.set_hop(-s ,1,1,[ 0, 1, 1]) #k_y + k_z#
    TrNSmK32.set_hop(-s ,1,1,[ 0, -1, 1])   #k_z - k_y#
    TrNSmK32.set_hop(-s ,1,1,[ 1, 0, 1]) #k_x + k_z#
    TrNSmK32.set_hop(-s ,1,1,[-1, 0, 1])   #k_z - k_x#

    # Terminos fuera de la diagonal a_1
    ## intracell hopping
    TrNSmK32.set_hop(-1-m0**2 ,0,1,[ 0, 0, 0]) 
    ##
    TrNSmK32.set_hop(m0 - 2*s ,0,1,[ 1, 0, 0])
    TrNSmK32.set_hop(m0 + 2*s ,0,1,[ -1, 0, 0])
    TrNSmK32.set_hop(m0 ,0,1,[ 0, 1, 0])
    TrNSmK32.set_hop(m0 ,0,1,[ 0,-1, 0])
    TrNSmK32.set_hop(m0 ,0,1,[ 0, 0, 1])
    TrNSmK32.set_hop(m0 ,0,1,[ 0, 0,-1]) 
    ##  2k...
    TrNSmK32.set_hop(-0.5*(c+s),0,1,[2 , 0, 0])
    TrNSmK32.set_hop(-0.5*(c-s),0,1,[-2 , 0, 0])
    TrNSmK32.set_hop(-0.5*c ,0,1,[0 , 2, 0])
    TrNSmK32.set_hop(-0.5*c ,0,1,[0 , -2, 0])
    TrNSmK32.set_hop( -c ,0,1,[0 , 0, 2])
    TrNSmK32.set_hop( -c ,0,1,[0 , 0,-2])
    ## cos(k_l+ k_m)
    TrNSmK32.set_hop(-c + 1.5*s,0,1,[ 1, 1, 0])
    TrNSmK32.set_hop(-c - 1.5*s ,0,1,[ -1, -1, 0]) ## cos k_x+k_y
    TrNSmK32.set_hop(-c + 1.5*s,0,1,[ 1, -1, 0])
    TrNSmK32.set_hop(-c- 1.5*s ,0,1,[ -1, 1, 0])  ## cos k_x-k_y
    TrNSmK32.set_hop(-c ,0,1,[ 1, 0, 1])
    TrNSmK32.set_hop(-c ,0,1,[ -1, 0, -1]) ## cos k_x+k_z
    TrNSmK32.set_hop(-c ,0,1,[ 1,0 , -1])
    TrNSmK32.set_hop(-c ,0,1,[-1, 0, 1])  ## cos k_x-k_z
    TrNSmK32.set_hop(-c ,0,1,[ 0, 1, 1])
    TrNSmK32.set_hop(-c ,0,1,[ 0, -1, -1])  ## cos k_y+k_z
    TrNSmK32.set_hop(-c ,0,1,[ 0, -1, 1])
    TrNSmK32.set_hop(-c ,0,1,[ 0, 1, -1])  # cos k_y-k_z
    return TrNSmK32


# In[6]:


get_ipython().run_cell_magic('time', '', '# Generamos los puntos que nos dan Energía cero en nuestro hamiltoniano continuo\n# Exploración de las energías de Fermi para E=0,con distinto parametro m0\nNN = 301\nm0 = [1, 1.2, 2.8, 3, 3.2]\nmesh = []\nl=0\nfor i in range(len(m0)):\n    mesh1=Nodos(201 ,m0[i], 0)\n    if ( len( mesh1) != 0):\n        mesh1tb = mesh1 / (2*pi)\n        mesh.append(mesh1tb)\n        l+=1\n        \n# Generamos un subplot de los nodos en el espacio de momentos\nfig = plt.figure(figsize=(20, 4), dpi=100)\nfig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)\nfig.suptitle(\'Transiciones en el modelo de Nudo de Trebol de Líneas Nodales\', fontsize = 18)\nfor i in range(l):\n    i_1= i + 1\n    ax = fig.add_subplot(1, l , i_1, projection=\'3d\')\n    ax.scatter( mesh1tb[i].T[0], mesh1tb[i].T[1], mesh1tb[i].T[2])\n    ax.set_title(f\'$m_0$={m0[i]}\')\n    ax.set_xlabel(\'$k_x$\', fontsize = 14)\n    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_xticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_ylabel(\'$k_y$\', fontsize = 14)\n    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_yticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.set_zlabel(\'$k_z$\', fontsize = 14)\n    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])\n    ax.set_zticklabels([r\'$-\\pi$\', r\'$-\\pi/2$\', r\'0\', r\'$\\pi/2$\', r\'$\\pi$\'])\n    ax.view_init(20,-45)\nfig.savefig("TransicionesTrNLsinMasa.pdf")')


# In[ ]:


#Proyecciones de las superficies de Fermi en los planos canonicos 
fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
X=[x, y, z]
l = len(X)
for i in range( l):
    for j in range( l):
        k=0
        if (j > i ):
            ax = fig.add_subplot(1, l, k)
            ax.scatter( mesh1tb[0].T[i],mesh1tb[0].T[j])
            ax.set_xlabel(f'$k_{X[i]}$', fontsize = 14)
            ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
            ax.set_ylabel(f'$k_{X[j]}$', fontsize = 14)
            ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
            k+=1

fig.savefig("FBSTrNL_m01_mz0.png", bbox_inches ='tight')
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Generamos los puntos que nos dan Energía cero en nuestro hamiltoniano continuo\n# Se varia el termino de masa\nNN = 301\nmesh12= []\nmz12 = [0, 0.07, 0.1, 0.2]\nfor i in range(len(mz12)):\n    mesh12.append(Nodos(201, 1.2,mz12[i]))\n    \nmesh12tb = array(mesh12)\nmesh12tb = mesh12tb / (2*pi)')


# In[ ]:


#Exploracion de E=0 con el parametro de masa 
fig = plt.figure(figsize=(20, 4), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
fig.suptitle('Transiciones en el modelo de Nudo de Trebol con $m_0=1.2$ y m_z', fontsize = 18)
#---- First subplot
l = len(mz12)
for i in range(l):
    i_1= i + 1
    ax = fig.add_subplot(1, l , i_1, projection='3d')
    ax.scatter( mesh12tb[i].T[0], mesh12tb[i].T[1], mesh12tb[i].T[2])
    ax.set_title(f'$m_0$={mz12[i]}')
    ax.set_xlabel('$k_x$', fontsize = 14)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('$k_y$', fontsize = 14)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_zlabel('$k_z$', fontsize = 14)
    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_zticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.view_init(20,-45)
fig.savefig("TransicionesTrNLconMasa12.pdf")


# In[ ]:


#Exploracion de las superficies con las proyecciones en algun plano
fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh12tb[0].T[0],mesh12tb[0].T[1])
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh12tb[0].T[0],mesh12tb[0].T[2])
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh12tb[0].T[1],mesh12tb[0].T[2])
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m012_mz0.png", bbox_inches ='tight')
plt.show()


# In[ ]:


for j in range(1,len(mz12)):

    fig = plt.figure(figsize=(15, 5), dpi=100)
    fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter( mesh12tb[j].T[0],mesh12tb[j].T[1])
    ax.set_xlabel('$k_x$', fontsize = 14)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('$k_y$', fontsize = 14)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter( mesh12tb[j].T[0],mesh12tb[j].T[2])
    ax.set_xlabel('$k_x$', fontsize = 14)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('$k_z$', fontsize = 14)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

    ax = fig.add_subplot(1, 3, 3)
    ax.scatter( mesh12tb[j].T[1],mesh12tb[j].T[2])
    ax.set_xlabel('$k_y$', fontsize = 14)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('$k_z$', fontsize = 14)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    fig.savefig(f'FBSTrNL_m01_mz{mz12[j]}.png', bbox_inches ='tight')
    plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh1tb[3].T[0],mesh1tb[3].T[1],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh1tb[3].T[0],mesh1tb[3].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh1tb[3].T[1],mesh1tb[3].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m03_mz0.png",bbox_inches ='tight')
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'mesh2= []\nmz = [0.045, 0.058, 0.060, 0.070]\nfor i in range(len(mz)):\n    mesh2.append(Nodos(201 ,2.8,mz[i]))\n    \nmesh2tb = array(mesh2)\nmesh2tb = mesh2tb / (2*pi)')


# In[ ]:


fig = plt.figure(figsize=(20, 4), dpi=100)
fig.suptitle('Transiciones en el modelo nudo trebol nodal con un término de masa $m_z$ y $m_0=2.8$', fontsize = 18)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
l = len(mz)
for i in range(l):
    i_1= i + 1
    ax = fig.add_subplot(1, l , i_1, projection='3d')
    ax.scatter( mesh2tb[i].T[0],mesh2tb[i].T[1],mesh2tb[i].T[2] )
    ax.set_title(f'$m_z$={mz[i]}')
    ax.set_xlabel('$k_x$', fontsize = 14)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('$k_y$', fontsize = 14)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.set_zlabel('$k_z$', fontsize = 14)
    ax.set_zticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_zticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    ax.view_init(80,-50)
fig.savefig("TransicionesLNLconMasa.pdf")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh1tb[2].T[0],mesh1tb[2].T[1],s=5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh1tb[2].T[0],mesh1tb[2].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh1tb[2].T[1],mesh1tb[2].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m028_mz0.png",bbox_inches ='tight')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh2tb[0].T[0],mesh2tb[0].T[1],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh2tb[0].T[0],mesh2tb[0].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh2tb[0].T[1],mesh2tb[0].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m028_mz045.png",bbox_inches ='tight')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh2tb[1].T[0],mesh2tb[1].T[1],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh2tb[1].T[0],mesh2tb[1].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh2tb[1].T[1],mesh2tb[1].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m028_mz058.png",bbox_inches ='tight')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh2tb[2].T[0],mesh2tb[2].T[1],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh2tb[2].T[0],mesh2tb[2].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$')
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh2tb[2].T[1],mesh2tb[2].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m028_mz060.png",bbox_inches ='tight')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 5), dpi=100)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
ax = fig.add_subplot(1, 3, 1)
ax.scatter( mesh2tb[3].T[0],mesh2tb[3].T[1],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_y$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 2)
ax.scatter( mesh2tb[3].T[0],mesh2tb[3].T[2],s = 5)
ax.set_xlabel('$k_x$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])

ax = fig.add_subplot(1, 3, 3)
ax.scatter( mesh2tb[3].T[1],mesh2tb[3].T[2],s = 5)
ax.set_xlabel('$k_y$', fontsize = 14)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
ax.set_ylabel('$k_z$', fontsize = 14)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
fig.savefig("FBSTrNL_m028_mz07.png",bbox_inches ='tight')
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Construimos los sistemas de amarre fuerte por analizar\nWeylNL1 = make_tbsyst(1, 1, 0)\nWeylNL3 = make_tbsyst(1, 3, 0)\n\nWeylNL28 = make_tbsyst(1, 2.8,0)\nWeylNL28_mz0045 = make_tbsyst(1, 2.8, 0.045)\nWeylNL28_mz0058 = make_tbsyst(1, 2.8, 0.058)\nWeylNL28_mz006 = make_tbsyst(1, 2.8, 0.06)\nWeylNL28_mz007 = make_tbsyst(1, 2.8,0.07)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'proc = Pool()\n# Calculo de las estructuras de Banda\nWeyl_finito_z = WeylNL1.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl1_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl1_1 = array(Evnl1_1)\nEvnl1_1 = Evnl1_1.T\nfile = open("TrNEvlm0_1_mz_0_1.txt","w")\nfor En in Evnl1_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL1.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl1_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl1_2 = array(Evnl1_2)\nEvnl1_2 = Evnl1_2.T\nfile = open("TrNEvlm0_1_mz_0_2.txt","w")\nfor En in Evnl1_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL1.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl1_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl1_3 = array(Evnl1_3)\nEvnl1_3 = Evnl1_3.T\nfile = open("TrNEvlm0_1_mz_0_3.txt","w")\nfor En in Evnl1_3:\n    np.savetxt(file, En)\nfile.close()\n\n###################################################################\nWeyl_finito_z = WeylNL3.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl3_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl3_1 = array(Evnl3_1)\nEvnl3_1 = Evnl3_1.T\nfile = open("TrNEvlm0_3_mz_0_1.txt","w")\nfor En in Evnl3_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL3.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl3_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl3_2 = array(Evnl3_2)\nEvnl3_2 = Evnl3_2.T\nfile = open("TrNEvlm0_3_mz_0_2.txt","w")\nfor En in Evnl3_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL3.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl3_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl3_3 = array(Evnl3_3)\nEvnl3_3 = Evnl3_3.T\nfile = open("TrNEvlm0_3_mz_0_3.txt","w")\nfor En in Evnl3_3:\n    np.savetxt(file, En)\nfile.close()\n####################################################################\nWeyl_finito_z = WeylNL28.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl28_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl28_1 = array(Evnl28_1)\nEvnl28_1 = Evnl28_1.T\nfile = open("TrNEvlm0_28_mz_0_1.txt","w")\nfor En in Evnl28_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL28.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl28_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl28_2 = array(Evnl28_2)\nEvnl28_2 = Evnl28_2.T\nfile = open("TrNEvlm0_28_mz_0_2.txt","w")\nfor En in Evnl28_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL28.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl28_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl28_3 = array(Evnl28_3)\nEvnl28_3 = Evnl28_3.T\nfile = open("TrNEvlm0_28_mz_0_3.txt","w")\nfor En in Evnl28_3:\n    np.savetxt(file, En)\nfile.close()\n###################################################################\nWeyl_finito_z = WeylNL28_mz0045.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl28_2_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl28_2_1 = array(Evnl28_2_1)\nEvnl28_2_1 = Evnl28_2_1.T\nfile = open("TrNEvlm0_28_mz_02_1.txt","w")\nfor En in Evnl28_2_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL28_mz0045.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl28_2_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl28_2_2 = array(Evnl28_2_2)\nEvnl28_2_2 = Evnl28_2_2.T\nfile = open("TrNEvlm0_28_mz_02_2.txt","w")\nfor En in Evnl28_2_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL28_mz0045.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl28_2_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl28_2_3 = array(Evnl28_2_3)\nEvnl28_2_3 = Evnl28_2_3.T\nfile = open("TrNEvlm0_28_mz_02_3.txt","w")\nfor En in Evnl28_2_3:\n    np.savetxt(file, En)\nfile.close()\n###################################################################\nWeyl_finito_z = WeylNL28_mz0058.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl28_4_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl28_4_1 = array(Evnl28_4_1)\nEvnl28_4_1 = Evnl28_4_1.T\nfile = open("TrNEvlm0_28_mz_04_1.txt","w")\nfor En in Evnl28_4_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL28_mz0058.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl28_4_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl28_4_2 = array(Evnl28_4_2)\nEvnl28_4_2 = Evnl28_4_2.T\nfile = open("TrNEvlm0_28_mz_04_2.txt","w")\nfor En in Evnl28_4_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL28_mz0058.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl28_4_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl28_4_3 = array(Evnl28_4_3)\nEvnl28_4_3 = Evnl28_4_3.T\nfile = open("TrNEvlm0_28_mz_04_3.txt","w")\nfor En in Evnl28_4_3:\n    np.savetxt(file, En)\nfile.close()')


# In[ ]:


###################################################################
Weyl_finito_z = WeylNL28_mz006.cut_piece( 200, 2,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)
Evnl28_6_1 = proc.map(Weyl_finito_z.solve_one,k_vec)
Evnl28_6_1 = array(Evnl28_6_1)
Evnl28_6_1 = Evnl28_6_1.T
file = open("TrNEvlm0_28_mz_06_1.txt","w")
for En in Evnl28_6_1:
    np.savetxt(file, En)
file.close()

Weyl_finito_y = WeylNL28_mz006.cut_piece( 200, 1,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)
Evnl28_6_2 = proc.map(Weyl_finito_y.solve_one,k_vec)
Evnl28_6_2 = array(Evnl28_6_2)
Evnl28_6_2 = Evnl28_6_2.T
file = open("TrNEvlm0_28_mz_06_2.txt","w")
for En in Evnl28_6_2:
    np.savetxt(file, En)
file.close()

Weyl_finito_x = WeylNL28_mz006.cut_piece( 200, 0,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)
Evnl28_6_3 = proc.map(Weyl_finito_x.solve_one,k_vec)
Evnl28_6_3 = array(Evnl28_6_3)
Evnl28_6_3 = Evnl28_6_3.T
file = open("TrNEvlm0_28_mz_06_3.txt","w")
for En in Evnl28_6_3:
    np.savetxt(file, En)
file.close()
###################################################################
Weyl_finito_z = WeylNL28_mz007.cut_piece( 200, 2,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)
Evnl28_7_1 = proc.map(Weyl_finito_z.solve_one,k_vec)
Evnl28_7_1 = array(Evnl28_7_1)
Evnl28_7_1 = Evnl28_7_1.T
file = open("TrNEvlm0_28_mz_07_1.txt","w")
for En in Evnl28_7_1:
    np.savetxt(file, En)
file.close()

Weyl_finito_y = WeylNL28_mz007.cut_piece( 200, 1,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)
Evnl28_7_2 = proc.map(Weyl_finito_y.solve_one,k_vec)
Evnl28_7_2 = array(Evnl28_7_2)
Evnl28_7_2 = Evnl28_7_2.T
file = open("TrNEvlm0_28_mz_07_2.txt","w")
for En in Evnl28_7_2:
    np.savetxt(file, En)
file.close()

Weyl_finito_x = WeylNL28_mz007.cut_piece( 200, 0,glue_edgs=False )
path = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]
k_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)
Evnl28_7_3 = proc.map(Weyl_finito_x.solve_one,k_vec)
Evnl28_7_3 = array(Evnl28_7_3)
Evnl28_7_3 = Evnl28_7_3.T
file = open("TrNEvlm0_28_mz_07_3.txt","w")
for En in Evnl28_7_3:
    np.savetxt(file, En)
file.close()


# In[ ]:


#Graficamos las estructuras de Banda
label = [   r"$-X$",r"$\Gamma$", r"$X$"]
label2 = [   r"$-Y$",r"$\Gamma$", r"$Y$"]
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=3$ y $m_z=0$', fontsize=14)
for n in range(len(Evnl3_1 )):
    ax[0].plot(k_dist,Evnl3_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-3,3)
############################################################
for n in range(len(Evnl3_2 )):
    ax[1].plot(k_dist,Evnl3_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-3,3)
###############################################################
for n in range(len(Evnl3_3 )):
    ax[2].plot(k_dist,Evnl3_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-3,3)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm0_3_mz0.pdf")
##############################################################
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=1$ y $m_z=0$', fontsize=14)
for n in range(len(Evnl1_1 )):
    ax[0].plot(k_dist,Evnl1_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-3,3)
############################################################
for n in range(len(Evnl1_2 )):
    ax[1].plot(k_dist,Evnl1_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-3,3)
###############################################################
for n in range(len(Evnl1_3 )):
    ax[2].plot(k_dist,Evnl1_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-3,3)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm0_1_mz0.pdf")


# In[ ]:


fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=2.8$ y $m_z=0$', fontsize=14)
for n in range(len(Evnl28_1 )):
    ax[0].plot(k_dist,Evnl28_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl28_2 )):
    ax[1].plot(k_dist,Evnl28_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl28_3 )):
    ax[2].plot(k_dist,Evnl28_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm0_28_mz0.pdf")
##################################################################
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=2.8$ y $m_z=0.045$', fontsize=14)
for n in range(len(Evnl28_2_1 )):
    ax[0].plot(k_dist,Evnl28_2_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl28_2_2 )):
    ax[1].plot(k_dist,Evnl28_2_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl28_2_3 )):
    ax[2].plot(k_dist,Evnl28_2_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm028_mz_0045.pdf")


# In[ ]:


fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=2.8$ y $m_z=0.058$', fontsize=14)
for n in range(len(Evnl28_4_1 )):
    ax[0].plot(k_dist,Evnl28_4_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl28_4_2 )):
    ax[1].plot(k_dist,Evnl28_4_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl28_4_3 )):
    ax[2].plot(k_dist,Evnl28_4_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm028_mz0058.pdf")
########################################################
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=2.8$ y $m_z=0.06$', fontsize=14)
for n in range(len(Evnl28_6_1 )):
    ax[0].plot(k_dist,Evnl28_6_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl28_6_2 )):
    ax[1].plot(k_dist,Evnl28_6_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl28_6_3 )):
    ax[2].plot(k_dist,Evnl28_6_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm028_mz006.pdf")
########################################################
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=2.8$ y $m_z=0.07$', fontsize=14)
for n in range(len(Evnl28_7_1 )):
    ax[0].plot(k_dist,Evnl28_7_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl28_7_2 )):
    ax[1].plot(k_dist,Evnl28_7_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl28_7_3 )):
    ax[2].plot(k_dist,Evnl28_7_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm028_mz007.pdf")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'WeylNL12_mz0045 = make_tbsyst(1, 1.2, 0.045)\nproc = Pool()\nWeyl_finito_z = WeylNL12_mz0045.cut_piece( 200, 2,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_z.k_path(path,401,report=False)\nEvnl12_2_1 = proc.map(Weyl_finito_z.solve_one,k_vec)\nEvnl12_2_1 = array(Evnl12_2_1)\nEvnl12_2_1 = Evnl12_2_1.T\nfile = open("TrNEvlm0_12_mz_02_1.txt","w")\nfor En in Evnl12_2_1:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_y = WeylNL12_mz0045.cut_piece( 200, 1,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_y.k_path(path,401,report=False)\nEvnl12_2_2 = proc.map(Weyl_finito_y.solve_one,k_vec)\nEvnl12_2_2 = array(Evnl12_2_2)\nEvnl12_2_2 = Evnl12_2_2.T\nfile = open("TrNEvlm0_12_mz_02_2.txt","w")\nfor En in Evnl12_2_2:\n    np.savetxt(file, En)\nfile.close()\n\nWeyl_finito_x = WeylNL12_mz0045.cut_piece( 200, 0,glue_edgs=False )\npath = [[-0.5,0.0],[0.0,0.0],[0.5,0.0]]\nk_vec, k_dist, k_node = Weyl_finito_x.k_path(path,401,report=False)\nEvnl12_2_3 = proc.map(Weyl_finito_x.solve_one,k_vec)\nEvnl12_2_3 = array(Evnl12_2_3)\nEvnl12_2_3 = Evnl12_2_3.T\nfile = open("TrNEvlm0_12_mz_02_3.txt","w")\nfor En in Evnl12_2_3:\n    np.savetxt(file, En)\nfile.close()')


# In[ ]:


label = [   r"$-X$",r"$\Gamma$", r"$X$"]
label2 = [   r"$-Y$",r"$\Gamma$", r"$Y$"]
##############################################################
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
fig.tight_layout(pad=2.0)
fig.suptitle('Espectro de bandas para $m_0=1.2$ y $m_z=0.045$', fontsize=14)
for n in range(len(Evnl12_2_1 )):
    ax[0].plot(k_dist,Evnl12_2_1[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[0].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[0].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[0].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[0].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[0].set_ylim(-0.5,0.5)
############################################################
for n in range(len(Evnl12_2_2 )):
    ax[1].plot(k_dist,Evnl12_2_2[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[1].set_xlabel("Camino en el espacio $k_x$")
# Colocamos una etiqueta al eje y
ax[1].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[1].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[1].set_xticklabels(label)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[1].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[1].set_ylim(-0.5,0.5)
###############################################################
for n in range(len(Evnl12_2_3 )):
    ax[2].plot(k_dist,Evnl12_2_3[n],'-k',alpha=0.2)

# Colocamos una etiqueta al eje x
ax[2].set_xlabel("Camino en el espacio $k_y$")
# Colocamos una etiqueta al eje y
ax[2].set_ylabel("Energía")
# Colocamos los xticks en los puntos del path
ax[2].set_xticks(k_node)
# Colocamos las etiquetas de los xticks 
ax[2].set_xticklabels(label2)

# Dibujamos líneas verticales en cada xtick
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], lw=0.5, color='k')

# Especificamos los límites de graficación en el 
# eje de las abcisas
ax[2].set_ylim(-0.5,0.5)

# Ajustamos los ejes y etiquetas antes de guardar la figura
fig.tight_layout()
# Guardamos la figura como un pdf
fig.savefig("BSTrNLm0_12_mz0045_2.pdf")


# In[ ]:




