# Multi_Face_Detector Projet 5 année en Intelligence Artificielle.

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

![](https://github.com/MatD3mons/Multi_Face_Detector/blob/main/Image/Capture.PNG)
__Nous avons donc choisi de réaliser un Faster-RCNN de par sa vitesse de prédiction bien plus raisonnable qu'un simple RCNN par exemple.__

## Notre Dataset
* https://github.com/MatD3mons/Multi_Face_Detector/tree/main/Train


## Quelques points techniques
#### le ROI Pooling
* https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af

* https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44

#### Différents temps de prédictions
*Temps de prédiction sous gpu*
![](https://github.com/MatD3mons/Multi_Face_Detector/blob/main/Image/detect.PNG)

*__Je te laisse mettre le temps de prédictions de plusieurs images stp, pour bien montrer qu'on a un faster-RCNN, merci__*
Pour le dectector cv2
![](https://github.com/MatD3mons/Multi_Face_Detector/blob/main/Image/photo%20CV2%20face.PNG)

Pour le Faster RCNN
![](https://github.com/MatD3mons/Multi_Face_Detector/blob/main/Image/photo%20Faster%20RCNN.PNG)

#### Comment utiliser notre serveur pour faire vos prédictions facilement
* https://github.com/MatD3mons/Multi_Face_Detector/tree/main/Serveur

#### Améliorations possible
* Un meilleur entraînement du modèle, actuellement notre modèle s'est entraîné sur 3300 photos et 800 pour la validation (comme dit précédemment, notre but était d'avoir un modèle fonctionnel et non "parfait", de plus une époque durait plus d'une heure, ce n'était qu'une question de gestion de temps qui nous a poussé à ne pas utiliser tout le dataset (mais également de mémoire pour Matthieu), sachant qu'il y a plus de 10000 photos sur notre dataset, l'entraîner sur l'intégralité du dataset pourrait être une option d'amélioration.
* Utiliser un réseau resnet ayant une meilleure précision, mais comme le point au dessus l'a précisé, ce n'était qu'un manque de temps.
* Pour aller plus loin, nous pourrions réaliser de l'instance segmentation en utilisant un MASK RCNN en utilisant un algorithme pour récupérer une des data manquante pour entraîner le modèle si nous gardons ce dataset. (Plus d'information car ce type de detection ne sera pas expliqué dans ce readme : https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/), ce qui permettrait d'avoir une meilleure précision sur la localisation d'un visage dans une image (et non pas juste les coordonnées des bounding boxs) pour ensuite pourquoi pas, utiliser le modèle dans un projet plus ambitieux.

Deep Face Recognition
British Machine Vision Conference, 2015.
