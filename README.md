# DoodleAnimals

### Tim:
Boško Kulušić E2 24/2024
Jelena Kovač  E2 14/2024

### Definicija problema:
Projektom se rešava problem klasifikacije gde se prepoznaju ručno nacrtane skice raznih životinja. Izabrali smo 20 klasa i sistem će prepoznavati koja se nalazi na slici. U pitanju su sledeće životinje: cat(mačka), cow(krava), crocodile(krokodil), dog(pas), duck(patka), elephant(slon), fish(riba), hedgehog(jež), horse(konj), kangaroo(kengur), lion(lav), monkey(majmun), owl(sova), panda(panda), pig(svinja), sheep(ovca), snail(puž), snake(zmija), spider(pauk), zebra(zebra).

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
