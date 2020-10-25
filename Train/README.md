# Notre Dataset

* Disponible sur https://www.robots.ox.ac.uk/~vgg/data/vgg_face/ Notre Dataset contient 15000+ photos de base de visages de célébrités, ainsi que les coordonnées des boundings boxes de chaques photos. Une fois toutes les photos téléchargés via un script, il a fallut nettoyer le dataset, nous avons 2 algorithmes pour cela, un qui se base sur un algorithme de reconnaissance de visage qui supprimera les photos sans visages/corrompu, et un autre qui se contente juste de supprimer les photos corrompu. *__Explique comment sont organisés les labels .csv et Insère un screen de toes scripts stp, faut un screen du code qui clean, et un autre pour dl la data. Si tu peux mets un ptit screen du show batch de quelques photos pour qu'on puisse voir à quoi ça ressemble stp, moi jpeux pas y'a plus rien qui fonctionne je crois j'ai delete mon dossier sur mon drive et flemme de tout redl déso__*  

## Quelques points techniques
#### le ROI Pooling, c'est quoi ?
*__Si t'arrives à trouver un lien qui explique vraiment bien pq on l'utilise je suis chaud pcq ça me sule là je retrouve plus mon lien__*
Plus d'information : https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af
ou : https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44

#### Architecture de réseau
```
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```
Notre choix d'architecture était assez mitigé entre un réseau resnet ou un reseau vgg, cependant après plusieurs recherches, notre choix pour le resnet50 a été confirmé. un réseau vgg16 ou vgg19 est plus long à entraîner qu'un resnet50 ou resnet101, de plus un resnet 50 fait preuve d'une meilleur précision comparé au vgg16/vgg19. Notre but n'étant pas de faire un modèle parfait mais fonctionnel et pouvant faire des prédictions correcte, nous nous sommes donc basé sur un réseau resnet50 pré-entrainé sur le dataset coco.

Plus d'informations sur les performances de différentes architectures de réseau : https://github.com/jcjohnson/cnn-benchmarks

Dataset source : [1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.
