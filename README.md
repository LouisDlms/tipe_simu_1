# TIPE - Simulation n°1

**************************************************
*           GUIDE RAPIDE D'UTILISATION           *
**************************************************

I - Généralités

    Ce programme permet de simuler le diffusion d'ondes de surface dans une cuve à onde à travers une structure de type "cape d'invisibilité". Le calcul s'effectue par une évolution par petits pas temporels d'une version discrétisée dans un maillage de la cuve à onde. Le calcul peut s'effectuer, selon votre convenance, sur CPU ou sur GPU.


II - Paramètres de la simulation

    Les divers paramètres sont (en unité SI ou sans dimension):
        - Npt : le nombre de points du côté du maillage.
        - L : le côté de la cuve à onde (carrée)
    Ces deux paramètres définissent donc le pas spatial.
    
        - dt : le pas temporel de la simulation. Valeur conseillée : 0.00001
        - T : la durée totale de la simulation
        
        - r0 : le rayon intérieur de la structure
        - Nrg : Nombre de rangs de plots dans la cape
    /!\ Ces deux paramètres doivent être choisis afin que la structure ne dépasse pas de la cuve à onde
        - Npl : Nombre plots par rangs de la cape
    
        - periode : période de l'onde produite par la cuve à onde
        - A : paramètre proportionnel à l'amplitude de l'onde produite par la cuve
    
        - tau : durée entre deux enregistrements de l'état d'évolution du système (sous forme d'image, dans le répertoire courant)
        - dimension : 2 ou 3, correspond à la nature de l'enrgistrement, vue 2d du dessus ou vue 3d. Le calcul d'une vue 3d est notablement plus long.


III - Calcul sur CPU ou GPU
    
    Si le paramètre Npt choisis est élevé (plus que de l'ordre de la centaine), l'utilisation du calcul parrallèle accéléré sur carte graphique induit une amélioration conséquente des performances, d'environ un facteur 15 (en tout cas, avec un i5-7200U comparé à une gtx 940MX).
    Le passage sur GPU nécessite :
        - Une carte graphique performante compatible Nvidia CUDA (cf liste ici : https://developer.nvidia.com/cuda-gpus )
        - L'installation de CUDA (disponible ici : https://developer.nvidia.com/cuda-downloads )
        - L'installation du module cupy sur python (cf. : https://docs-cupy.chainer.org/en/stable/install.html )
    /!\ La mémoire vive d'une carte graphique est généralement plus facilement saturée que celle du processeur central. Si des erreurs de ce type surviennent, il faut réduire le paramètre Npt


IV - C'est parti !

    Syntaxe : proggpu(Npt, Nrg, Npl, r0, dt, T, L, tau, periode, A, dimension)
    ou
    progcpu(Npt, Nrg, Npl, r0, dt, T, L, tau, periode, A, dimension)
