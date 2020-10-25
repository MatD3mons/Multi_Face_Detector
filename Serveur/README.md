# Multi_Face_Detector Projet 5 année en Intelligence Artificielle.

## Serveur

Avant de commencer, notre serveur ne tourne pas sous gpu, il possède une carte graphique non compatible avec cuda,
les prédictions prendront donc plus de temps (environ 6-8s) sans compter que le serveur en lui même est long,
compter à peu près 1min par prédiction, cependant le notebook reste disponible si vous voulez tester sous votre gpu.
Pour essayer avec une webcam, nous vous laissons regarder le code source disponible.

Avant de lancer votre serveur. il faut remplir un dossier "models" avec les différent technique rencontrer dans la partie "Train"
```
Models
- FasterRCNN
- haarcascade_frontalface_default.xml
- RCNN.h5
```

Une fois le dossier remplie vous pouvez lancer votre serveur en utilisant la commande suivant :
```
python api.py
```

Deep Face Recognition
British Machine Vision Conference, 2015.
