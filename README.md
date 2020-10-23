# Multi_Face_Detector Projet 5 année en Intelligence Artificielle.
*__Toutes les explications seront simplifiées pour éviter que le readme soit trop long__*

**Plan :**
  1. Différence entre CNN, RCNN, Fast RCNN et Faster RCNN (et pq on a choisi celui là plutot qu'un autre)
  2. On explique notre data (ce qui y'a, d'où elle vient, comment on l'a clean, comment on l'utilise dans le notebook)
  3. Quelques notions/bouts de code techniques
  4. Résultats de différents IA qu'on a (avec temps de prédiction et pq que lui il est moins ouf des trucs du genre)
  5. Tuto comment utiliser le serveur

## Différences entre CNN, RCNN, Fast-RCNN, Faster-RCNN
* Un CNN (Convolutionnal Neural Network) est un réseau de neuronne qiu permet avec une image en entrée du réseau, faire des prédictions sur l'objet dominant d'une image, c'est de la classification : Une image, un objet a prédire.

* Un RCNN (Region Based Convolutionnal Neural Network) est un réseau permettant de faire de la détection d'objets sur une image ou un flux vidéo (webcam par exemple) : une image, plusieurs objets potentiel pour la prédiction. le principe est simple, avec une image, un algorithme (selective search) vas proposer 2000 zones d'intérêts, et chaque zone d'intérêt sur l'image seront donnée en en entrée à la partie CNN du réseau, puis pour chaque zone d'intérêt, une prédiction sera faîte pour déterminer l'objet.

* Un Fast-RCNN est une "version" améliorée du RCNN, elle est plus rapide par le fait qu'au lieu de donner les 2000 zones d'intérêts en entrée à la partie CNN, cette fois-ci ce sera l'image qui sera donnée en entrée de la partie CNN, puis sur la feature map obtenue, l'algorithme selective search est également utilisée pour obtenir les zones d'intérêts qui seront ensuite resize grâce à un ROI Pooling (Region Of Interest Pooling, qui sera expliqué plus tard), puis les prédictions seront faites sur la feature map obtenue suite à ce ROI Pooling*

* Un Faster-RCNN est une version encore améliorée du Fast-RCNN, plus rapide par le fait que les zones d'intérêts sont obtenues grâce à un RPN (Region Proposal Network) qui prédira lui même. Ce qui résultera en un gain de temps.*

![](https://i.ytimg.com/vi/v5bFVbQvFRk/maxresdefault.jpg)

__Nous avons donc choisi de réaliser un Faster-RCNN de par sa vitesse de prédiction bien plus raisonnable qu'un simple RCNN par exemple.__

## Notre Dataset [1]
* Disponible sur https://www.robots.ox.ac.uk/~vgg/data/vgg_face/ Notre Dataset contient 15000+ photos de base de visages de célébrités, ainsi que les coordonnées des boundings boxes de chaques photos. Une fois toutes les photos téléchargés via un script, il a fallut nettoyer le dataset, nous avons 2 algorithmes pour cela, un qui se base sur un algorithme de reconnaissance de visage qui supprimera les photos sans visages/corrompu, et un autre qui se contente juste de supprimer les photos corrompu. *__Explique comment sont organisés les labels .csv et Insère un screen de toes scripts stp, faut un screen du code qui clean, et un autre pour dl la data. Si tu peux mets un ptit screen du show batch de quelques photos pour qu'on puisse voir à quoi ça ressemble stp, moi jpeux pas y'a plus rien qui fonctionne je crois j'ai delete mon dossier sur mon drive et flm de tout redl déso__*  

## Quelques points techniques

Plus d'information : https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af
#### le ROI Pooling, c'est quoi ?

# (*les screens pour éviter que je mette 3h à les retrouver pcq ma barre d'url est archi pleine j'en rajoute 10 par jours*)
 .*Architecture (simplifiée) (c'est pas le bon terme je sais je m'en occupe après)*
![](https://miro.medium.com/max/875/1*pSnVmJCyQIRKHDPt3cfnXA.png)

 .*Temps de prédiction de différents types de RCNN*
![](https://miro.medium.com/max/875/1*4gGddZpKeNIPBoVxYECd5w.png)

 .*ROI Pooling*
![](https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif.pagespeed.ce.5V5mycIRNu.gif)

![](https://d2l.ai/_images/roi.svg)

[1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.
