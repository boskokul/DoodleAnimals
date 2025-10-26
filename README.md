# DoodleAnimals
# Klasifikacija životinja na osnovu crteža primenom konvolutivnih neuronskih mreža

### Tim:
Boško Kulušić E2 24/2024
Jelena Kovač  E2 14/2024

### Definicija problema:
Projektom se rešava problem klasifikacije gde se prepoznaju ručno nacrtane skice raznih životinja. Izabrali smo 20 klasa i sistem će prepoznavati koja se nalazi na slici. U pitanju su sledeće životinje: cat(mačka), cow(krava), crocodile(krokodil), dog(pas), duck(patka), elephant(slon), fish(riba), hedgehog(jež), horse(konj), kangaroo(kengur), lion(lav), monkey(majmun), owl(sova), panda(panda), pig(svinja), sheep(ovca), snail(puž), snake(zmija), spider(pauk), zebra(zebra).

### Uputstvo za početak
U folderu **`finalno/`** se nalaze Jupiter Notebook fajlovi sa implementacijama različitih modela.
DesnseNet implementacija se nalazi u **`DenseNetFinal.ipynb`**, GoogleNet implementacija se nalazi
u **`GoogleNetfinalno.ipynb`**, ResNet implementacija se nalazi u **`ResNet_2lr.ipynb`**, 
EfficientNet implementacija se nalazi u *`EfficientNet_2lr.ipynb`**.

### Pokretanje
Jupiter Notebook fajlovi treba da se uvezu u **Google Colab** okruženje. Pored toga, na Google drive treba napraviti novi folder pod nazivom **dataset** gde je potrebno smestiti zipovan folder skupa podataka. Skup podataka je potrbeno nazvati **20animals**. 

* Podešavanje **Runtime** okruženja se radi na sledeći način:
	* Pod tabom **Runtime** pronaći **Change runtime type**
	* U otvorenom dijalogu, pod **Hardware accelerator** izabrati **T4 GPU**
	* Pod **Runtime Type** izabrati **Python 3**

Pokretanje Jupyter Notebook fajla se vrši na standardan način.

### Skup podataka:
U skupu podataka postoji 340 klasa skica (od kojih mi koristimo izabranih 20), svaka klasa sadrži po 3000 grayscale slika veličine 255x255 piksela u png formatu. 
Link do skupa podataka pod imenom Doodle Dataset: https://www.kaggle.com/datasets/ashishjangra27/doodle-dataset/data

### Metodologija:
Kako je skup podataka sređen i dobro balansiran, nisu potrebna velika pretprocesiranja niti augmentacija. Radićemo fine tuning poznatih CNN arhitektura pretreniranih nad ImageNet skupom, gde ćemo izlazni sloj prilagoditi našem problemu klasifikacije od 20 klasa. Kao arhitekture konvolutivnih neuronskih mreža odlučili smo se za:
 * Boško: DenseNet (Densely Connected Convolutional Networks) i GoogleNet(InceptionV1)
 * Jelena: ResNet(ResNetV2) i EfficientNet(EfficientNet-B3)
 
### Evaluacija:
Za evaluaciju koristićemo tačnost, F1, odziv i preciznost.
 * Podela skupa:
	* 42 000 slika (2 100 po klasi) za trening (70%)

	* 9 000 images (450 po klasi) za validacioni (15%)

	* 9 000 images (450 po klasi) za testni (15%)

### Primer rezultata (EfficienNetB3)
```txt
TEST SET REZULTATI
Accuracy:  90.18%
Precision: 90.28%
Recall:    90.18%
F1 Score:  90.17%

Po klasama:
              precision    recall  f1-score   support

         cat       0.93      0.90      0.91       450
         cow       0.83      0.81      0.82       450
   crocodile       0.94      0.91      0.92       450
         dog       0.66      0.62      0.64       450
        duck       0.90      0.92      0.91       450
    elephant       0.90      0.86      0.88       450
        fish       1.00      0.97      0.98       450
    hedgehog       0.95      0.93      0.94       450
       horse       0.80      0.89      0.84       450
    kangaroo       0.96      0.92      0.94       450
        lion       0.93      0.95      0.94       450
      monkey       0.87      0.88      0.87       450
         owl       0.95      0.90      0.92       450
       panda       0.89      0.90      0.90       450
         pig       0.81      0.91      0.86       450
       sheep       0.92      0.93      0.93       450
       snail       0.98      0.99      0.98       450
       snake       0.90      0.97      0.94       450
      spider       0.97      0.96      0.97       450
       zebra       0.97      0.92      0.95       450

    accuracy                           0.90      9000
   macro avg       0.90      0.90      0.90      9000
weighted avg       0.90      0.90      0.90      9000


Accuracy po klasama:
cat         : 89.56%
cow         : 81.11%
crocodile   : 90.89%
dog         : 62.00%
duck        : 91.56%
elephant    : 85.78%
fish        : 97.33%
hedgehog    : 92.67%
horse       : 89.33%
kangaroo    : 92.22%
lion        : 94.67%
monkey      : 87.78%
owl         : 90.00%
panda       : 90.22%
pig         : 90.89%
sheep       : 93.11%
snail       : 98.89%
snake       : 97.33%
spider      : 96.44%
zebra       : 91.78%

```