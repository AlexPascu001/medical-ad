# TODO - Medical Anomaly Detection

## 15.10.2025

- [x] mai multe ancore: repel intre ancore (imaginile "normale" - reprezentate de mai multe ancore)

- [x] pornire de la eigenfaces => le invatam si folosim toti termenii

- [x] dupa ce finalizam antrenarea => toate sample-urile normale sa se aglomereze in jurul unui anchor

- [x] anomalii finale => dist(sample test, cea mai apropiata ancora) 

- [x] calculez curba ROC si scot AUC (toate thresholdurile pozitive, true positive rate, etc)

- [x] metrici: image AUROC, pixel level AUROC

- [x] daca avem ancora => ne uitam la metrica generala (mse/distanta, pixel-wise test => anumite regiuni mai indepartate de ancora)

- [x] finetunez modelul + folosesc ancorele la test-time ca sa vedem cea mai apropiata ancora de imagine si la aceea ne referim

- [x] incercam sa venim cu niste ancore (cluster centroids, eigenfaces) (input space? embedding space?) care incearca sa traga sampleurile din jurul lor spre ele

- [x] cand ar fi o anomalie: ar tb sa nu fie aproape de nicio ancora

---

## 29.10.2025

- [x] varianta care optimizeaza distanta cosinus (1 - cos similarity) => la inferenta: ancorele prenormalizate, sa masori cos sim intre embd test si ancore si sa scor de anomalie 1 - max sim

- [ ] ancorele: learnable si atunci folosim cele 3 componente

- [ ] nr de ancore ca hiperparametru

- [x] baseline cu ancore random + cele cu k-means

- [ ] exploram cat de mult putem partea cu ancore 

- [ ] la urma putem adauga alte idei din medical AD 

- [ ] eu fac patch level nu pixel level, folosesc patch de la dino

- [ ] are sens sa ne uitam la patch level anchors??

- [ ] patchify image, pe fiecare imagine scoot ancore, patchurile

- [ ] (mai tarziu?) peste ancore sa aplicam un clasificator: si in loc de cel mai apropiat, sa facem un 1 vs all: o img se duce inspre ancora ca output de SVM in loc de cea mai apropiata (dupa ce vedem t-sne): daca se distribuie ok din tsne in jurul unei ancore sau nu

- [ ] cred ca in realitate am avea nevoie de mini decoder pentru scor de anomalii la nivel de pixel

- [x] sa fac o vizualizare cu t-sne pe embedding space: sa plotez ancorele, un subset de img normale si un subset de img anormale

- [ ] unele metode din BMAD genereaza pseudoanomalii, sa verific care

---

## 10.11.2025

- [ ] prima ancora + dupa alegem ancore cu probabilitate direct proportionala cu dist cosinus fata de ancorele alese
- [x] vizualizare t-sne cu aceleasi img pentru toate experimentele (random, kmeans, eigenface) si sa vedem cum se distribuie ancorele si img normale/anormale in embedding space
- [x] rulare 5 experimente pt random/kmeans/eigenface cu diferite seeduri si sa vedem variance in rezultate
- [ ] dupa ce vedem rezultatele cele mai bune, sa incercam ancorele learnable pt metoda potrivita
- [ ] 