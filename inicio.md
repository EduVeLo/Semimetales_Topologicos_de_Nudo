# Semimetales Topologicos de Nudo
En este jbook nosotros estudiamos un modelo de semimetales topológicos de nudo, a partir de un hamiltoniano de dos niveles en el espacio de momentos. Para entender los elementos del modelo, primero se realizará un mapeo de Hopf sobre la 3-esfera $\mathbb{S}^3$, lo que consiste en generar una esfera apartir de lazos cerrados, en el que solo se hará para una superficie toroidal, estos lazos corresponderán a los nudos toroidales. Entonces con los nudos toroidales propondremos nuestro modelo general para un hamiltoniano continuo que describa nodos de nudo, este modelo hereda la topología del toro y estos nodods describen un semimetal topológico de líneas nodales. Después, tomaremos los casos particulares de nuestros modelos continuos y propondremos un modelo de red, con el modulo pythtb de Python se explorarán las estructuras de bandas para nuestro modelo de banda de redes con condiciones de frontera abierta y se presentarán los estados de borde para cada uno de los modelos de red.

## Nudos Toroidales

Las líneas nodales surgen en la intersección de dos bandas, por ello consideremos un hamiltoniano de dos niveles asociada a dos bandas adyacentes,

\begin{align}
H(\mathbf{k})=a_0(\mathbf{k})\mathbb{I}+\mathbf{a}(\mathbf{k})\cdot\mathbf{\sigma} \quad \mathbf{\sigma}={\sigma_x,\sigma_y,\sigma_z}
\end{align}

La simetría $\mathcal{PT}$ en el hamiltoniano, impone que $H^*(\mathbf{k})=H(\mathbf{k})$, así en el modelo de dos bandas se tiene que $a_2(\mathbf{k})=0$, en adición, queremos que en el modelo las energías sean simétricos respecto a la energía cero lo que lleva $a_0(\mathbf{k})$=0. Nuestro modelo de dos bandas empleado es: 

$H(\mathbf{k})=a_1(\mathbf{k})\sigma_x+a_3(\mathbf{k})\sigma_z,$

la relación de dispersión de este hamilotniano es 

$E_s(\mathbf{k})=s\sqrt{a_1^2(\mathbf{k})+a_3^2(\mathbf{k})},\quad s=\pm,$

la generación de nodos sucede cuando la energía es $E=0$, donde las bandas s se tocan, lo que sucede con las condiciones: 

$a_1(\mathbf{k})=a_3(\mathbf{k})=0.$

Ahora introducimos una construcción genérica para la aproximación de modelos de nudos nodales, a partir del mapeos de Hopf.

Antes de presentar el modelo correspondiente al mapeo, definamos algunos de los conceptos a desarrollar.

Primero tomemos dos variables complejas $\mathit{z}$ y $\mathit{w}$, con la constricción $|\mathit{z}|^2+|\mathit{w}|^2=1$, lo cual describe una 3-esfera $\mathbb{S}^3$ (Esta proposición se observa si tomamos $\mathit{z=n_1+\mathbf{i} n_2}$ y $\mathit{w=n_3+\mathbf{i} n_4}$, 

$\mathit{(n_1+\mathbf{i} n_2)(n_1-\mathbf{i} n_2)+(n_3+\mathbf{i} n_4)(n_3-\mathbf{i} n_4)=n_1^2+n_2^2+n_3^2+n_4^2=1.} \quad )$

, consideremos la siguiente superficie sobre la 3-esfera

$|\mathit{z}|^p=|\mathit{w}|^q\quad (p,q)\in\mathbb{Z}^{2+} \quad \text{Enteros positivos,}$

esta superficie es topológicamente un 2-toro. Este hecho resulta de expresar nuestros numeros complejos con la identidad de euler$\mathit{z}=|\mathit{z}|e^{\mathbf{i}\theta_\mathit{z}}$ y $\mathit{w}=|\mathit{w}|e^{\mathbf{i}\theta_\mathit{w}}$, de esta forma, al costreñir nuestras variables complejas en la 3-esfera $|\mathit{z}|^2+|\mathit{w}|^2=1$ los modulos asociados a los números complejos queda fijo, esto es que:

\begin{align}
   |\mathit{z}|^2+|\mathit{w}|^2 &= 1  \\
   |\mathit{z}|^p &= |\mathit{w}|^q \\
\end{align}

para resolver este sistema de ecuaciones, la segunda ecuación lo elevamos al exponente $\frac{2}{p}$ y pasamos restando,

\begin{align}
  |\mathit{z}|^2+|\mathit{w}|^2 &= 1 \\
  |\mathit{z}|^2-|\mathit{w}|^{\frac{2q}{p}} &= 0; \\
  |\mathit{w}|^2 + |\mathit{w}|^{\frac{2q}{p}} &= 1;
\end{align}

así, los radios ahora están fijos y la superficie está parametrizada por las fases $\theta_\mathit{z}$ y $\theta_\mathit{w}$, que están asociados a la dirección toroidal y poloidal, respectivamente.

Dada las constricciones anteriores, proponemos una función de dos variables  complejas,$$\mathit{f(z,w)\equiv z^p+w^q=0,}$$, está función representa una superficie, si sobre está superficie le añadimos las dos constricciones anteriores, esto es equivalente a imponer una constricción sobre el toro, lo cual nos lleva a definir líneas, los cuales corresponden a los nudos toroidales, 
\begin{align}
	|\mathit{z}|^pe^{\I p\theta_\mathit{z}}+\mathit{w}^qe^{\I q\theta_\mathit{w}}&=0\quad \text{Expresando con la identidad de Euler}\\
|\mathit{z}|^p\big(e^{\I p\theta_\mathit{z}}+e^{\I q \theta_\mathit{w}}\big)&=0 \quad \text{donde}\quad|\mathit{z}|^p=|\mathit{w}|^q \\
e^{\I p \theta_\mathit{z}}&=-e^{\I q \theta_\mathit{w}}\quad \text{Multiplicamos por} \quad e^{-\I q \theta_\mathit{w}}\\
e^{\I p \theta_\mathit{z}-\I q \theta_\mathit{w}}&=-1=e^{-\I\pi}\\
 p \theta_\mathit{z}- q \theta_\mathit{w}&=\pi \quad \text{(mod $2\pi$)}
\end{align}
Esta ecuacion define nudos para los valores de p y q primos relativos, en otros casos forma links.

Con estos nudos tomamos los ceros de nuestra función $mathit{f(z,w)}$ para las energías de Fermi de nuestro modelo, la parte real e imaginaria de $\mathit{f}$ estarán asociados a nuestras dos funciones en nuestro modelo de bandas. Tomamos
$$a_1=\mathbb{R}e(f) \quad a_3=\mathbb{I}m(f)$$
