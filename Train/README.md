# Notre Dataset

* Disponible sur https://www.robots.ox.ac.uk/~vgg/data/vgg_face/

Une fois téléchargés il faut crée les fichier suivant:
```
files
Image
Label
Model
```

Notre Dataset contient 15000+ photos de base de visages de célébrités, ainsi que les coordonnées des boundings boxes de chaques photos. Une fois toutes les photos téléchargés via le notebook
```
INSTALL - DATA.ipynb
```
ou il a fallut nettoyer le dataset, nous avons un algorithmes pour cela, un qui se base sur un algorithme de reconnaissance de visage qui supprimera les photos sans visages/corrompu et qui se contente juste de supprimer les photos corrompu.

Une fois fait vous avez plusieur choix :
```
Faster RCNN - Pytorch.ipynb
```
qui permet d'entrainer un model de faster RCNN en pytorch
```
RCNN - Pytorch.ipynb
```
qui permet d'entrainer un model de RCNN en pytorch
```
RCNN - keras.ipynb
```
qui permet d'entrainer un model de RCNN en keras

une fois lancé les différent script vous pourriez les utiliser dans votre serveur en les déplacent dans le fichier :
```
Serveur/models
```
## Quelques points techniques
#### Architecture de réseau
```python
##Faster RCNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

##RCNN - Normal
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()

##RCNN - Edge
edge_detection = cv2.ximgproc.createStructuredEdgeDetection("model/model.yml")
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
edge_boxes.getBoundingBoxes(edges, orimap)

#Cv2
detector = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml');
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
faces = detector.detectMultiScale(gray, 1.3, 5);

```

Notre choix d'architecture était assez mitigé entre un réseau resnet ou un reseau vgg, cependant après plusieurs recherches, notre choix pour le resnet50 a été confirmé. un réseau vgg16 ou vgg19 est plus long à entraîner qu'un resnet50 ou resnet101, de plus un resnet 50 fait preuve d'une meilleur précision comparé au vgg16/vgg19. Notre but n'étant pas de faire un modèle parfait mais fonctionnel et pouvant faire des prédictions correcte, nous nous sommes donc basé sur un réseau resnet50 pré-entrainé sur le dataset coco.

Plus d'informations sur les performances de différentes architectures de réseau : https://github.com/jcjohnson/cnn-benchmarks

Dataset source : [1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015.
