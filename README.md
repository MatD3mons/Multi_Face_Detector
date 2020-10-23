# Multi_Face_Detector Projet 5 année en Intelligence Artificielle.
*__Toutes les explications seront simplifiées pour éviter que le readme soit trop long__*

**Plan :**
  1. Différence principale entre CNN, RCNN, Fast RCNN et Faster RCNN
  2. Notre Dataset
  3. Quelques notions/bouts de code techniques
  4. Résultats de différents IA que nous avons obtenus
  5. Comment utiliser le serveur pour faire des prédictions simplement, accessible à tout le monde
## Différences entre CNN, RCNN, Fast-RCNN, Faster-RCNN
* Un CNN (Convolutionnal Neural Network) est un réseau de neuronne qiu permet avec une image en entrée du réseau, faire des prédictions sur l'objet dominant d'une image, c'est de la classification : Une image, un objet a prédire.

* Un RCNN (Region Based Convolutionnal Neural Network) est un réseau permettant de faire de la détection d'objets sur une image ou un flux vidéo (webcam par exemple) : une image, plusieurs objets potentiel pour la prédiction. le principe est simple, avec une image, un algorithme (selective search) vas proposer 2000 zones d'intérêts, et chaque zone d'intérêt sur l'image seront donnée en en entrée à la partie CNN du réseau, puis pour chaque zone d'intérêt, une prédiction sera faîte pour déterminer l'objet.

* Un Fast-RCNN est une "version" améliorée du RCNN, elle est plus rapide par le fait qu'au lieu de donner les 2000 zones d'intérêts en entrée à la partie CNN, cette fois-ci ce sera l'image qui sera donnée en entrée de la partie CNN, puis sur la feature map obtenue, l'algorithme selective search est également utilisée pour obtenir les zones d'intérêts qui seront ensuite resize grâce à un ROI Pooling (Region Of Interest Pooling, qui sera expliqué plus tard), puis les prédictions seront faites sur la feature map obtenue suite à ce ROI Pooling.*

* Un Faster-RCNN est une version encore améliorée du Fast-RCNN, plus rapide par le fait que les zones d'intérêts sont obtenues grâce à un RPN (Region Proposal Network) qui prédira lui même. Ce qui résultera en un gain de temps.*

![](https://i.ytimg.com/vi/v5bFVbQvFRk/maxresdefault.jpg)

__Nous avons donc choisi de réaliser un Faster-RCNN de par sa vitesse de prédiction bien plus raisonnable qu'un simple RCNN par exemple.__

## Notre Dataset [1]
* Disponible sur https://www.robots.ox.ac.uk/~vgg/data/vgg_face/ Notre Dataset contient 15000+ photos de base de visages de célébrités, ainsi que les coordonnées des boundings boxes de chaques photos. Une fois toutes les photos téléchargés via un script, il a fallut nettoyer le dataset, nous avons 2 algorithmes pour cela, un qui se base sur un algorithme de reconnaissance de visage qui supprimera les photos sans visages/corrompu, et un autre qui se contente juste de supprimer les photos corrompu. *__Explique comment sont organisés les labels .csv et Insère un screen de toes scripts stp, faut un screen du code qui clean, et un autre pour dl la data. Si tu peux mets un ptit screen du show batch de quelques photos pour qu'on puisse voir à quoi ça ressemble stp, moi jpeux pas y'a plus rien qui fonctionne je crois j'ai delete mon dossier sur mon drive et flemme de tout redl déso__*  

## Quelques points techniques
#### le ROI Pooling, c'est quoi ?
*__Si t'arrives à trouver un lien qui explique vraiment bien pq on l'utilise je suis chaud pcq ça me sule là je retrouve plus mon lien__*
Plus d'information : https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af

#### Architecture de réseau
```
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```
Notre choix d'architecture était assez mitigé entre un réseau resnet ou un reseau vgg, cependant après plusieurs recherches, notre choix pour le resnet50 a été confirmé. un réseau vgg16 ou vgg19 est plus long à entraîner qu'un resnet50 ou resnet101, de plus un resnet 50 fait preuve d'une meilleur précision comparé au vgg16/vgg19. Notre but n'étant pas de faire un modèle parfait mais fonctionnel et pouvant faire des prédictions correcte, nous nous sommes donc basé sur un réseau resnet50 pré-entrainé sur le dataset coco.

Plus d'informations sur les performances de différentes architectures de réseau : https://github.com/jcjohnson/cnn-benchmarks

#### Différents temps de prédictions
![](https://cv-tricks.com/wp-content/uploads/2017/12/RCNN-speed-comparison.jpg)
     *Temps de prédiction sous gpu*
     
*__Je te laisse mettre le temps de prédictions de plusieurs images stp, pour bien montrer qu'on a un faster-RCNN, merci__*

#### Comment utiliser notre serveur pour faire vos prédictions facilement
Avant de commencer, notre serveur ne tourne pas sous gpu, il possède une carte graphique non compatible avec cuda, les prédictions prendront donc plus de temps (environ 6-8s) sans compter que le serveur en lui même est long, compter à peu près 1min par prédiction, cependant le notebook reste disponible si vous voulez tester sous votre gpu. Pour essayer avec une webcam, nous vous laissons regarder le code source disponible dans "/Serveur/Thread_Camera.py" pour ^le faire foncitonner.

*__Je te laisse mettre le screen d'une prédiction apres avoir expliqué comment y accéder__*

#### Améliorations possible

* Un meilleur entraînement du modèle, actuellement notre modèle s'est entraîné sur 3300 photos et 800 pour la validation (comme dit précédemment, notre but était d'avoir un modèle fonctionnel et non "parfait", de plus une époque durait plus d'une heure, ce n'était qu'une question de gestion de temps qui nous a poussé à ne pas utiliser tout le dataset (mais également de mémoire pour Matthieu), sachant qu'il y a plus de 10000 photos sur notre dataset, l'entraîner sur l'intégralité du dataset pourrait être une option d'amélioration.
* Utiliser un réseau resnet ayant une meilleure précision, mais comme le point au dessus l'a précisé, ce n'était qu'un manque de temps.
* Pour aller plus loin, nous pourrions réaliser de l'instance segmentation en utilisant un MASK RCNN en utilisant un algorithme pour récupérer une des data manquante pour entraîner le modèle si nous gardons ce dataset. (Plus d'information car ce type de detection ne sera pas expliqué dans ce readme : https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/), ce qui permettrait d'avoir une meilleure précision sur la localisation d'un visage dans une image (et non pas juste les coordonnées des bounding boxs) pour ensuite pourquoi pas, utiliser le modèle dans un projet plus ambitieux.

Dataset source : [1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.
