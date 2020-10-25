# Multi_Face_Detector Projet 5 année en Intelligence Artificielle.

## Serveur

Avant de commencer, notre serveur ne tourne pas sous gpu, il possède une carte graphique non compatible avec cuda,
les prédictions prendront donc plus de temps (environ 6-8s) sans compter que le serveur en lui même est long,
compter à peu près 1min par prédiction, cependant le notebook reste disponible si vous voulez tester sous votre gpu.
Pour essayer avec une webcam, nous vous laissons regarder le code source disponible dans :
```
"/Serveur/Thread_Camera.py"
```
pour le faire foncitonner.

Dataset source : [1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.
