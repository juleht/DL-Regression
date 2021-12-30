## Tulokset ja analyysi


# Datan muokkaus
Aineiston selittettävä muuttuja on kolumni (change of admit), todennäköisyys tulla valituksi yliopistoon. Selittäviin muuttujiin valikoidaan kaikki muut kolumnit paitsi indeksikolumni (Serial No.). Aineisto sisältää ainoastaan numeerisia kolumneja eikä siinä ole kategorisia muuttujia.

Aineisto jaetaan train set ja test settiin. Test_setin kooksi muodostuu 33 % aineistosta. Aineistossa on paljon eri skaalan muuttujia esim. GRE Score 0-340, jotta mallin optimointi-algoritmi asettaisi painotukset oikein muuttujat skaalataan välille 0-1.

# Malli
Kootaan varsinainen neuroverkko syväoppimista varten käyttämällä Sequential(). Tämä mahdollistaa neuroverkon kokoamisen kerros kerrokselta. Sequential-malli sallii vain yhden input-kerroksen ja yhden output-kerroksen. Neuroverkkoon valittiin yksi piilotettukerros kahdeksalla neuronilla. Piilotetulle layereille valitaan aktivaatio-funktio, joka on tässä tapauksessa on [ReLu (Rectified Linear unit)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). Käytetään mallin optimointi-algoritmina Adam-funktiota.

Mallin loss-funktioksi valitaan MSE (Mean squared error), jota algoritimi minimoi. MSE (Keskiarvoistettu neliö virhe) kuvastaa estimoitujen ja todellisten arvojen eroa toisistaan, sitä käytetään arvioimaan mallia, mitä pienempi virhe sen parempi malli. Lisäksi seurataan MAE (mean absolute error), joka antaa tarkemman kuvan siitä kuinka kaukana todellisista arvoista ollaan. MAE kertoo kuinka monta prosenttia todellisista arvoista malli on pielessä. 

Malli sovitetaan arvoilla epochs = 500 ja batch size = 128. Epochs määrittää kuinka monta iteraatiota datan läpi oppimisprosessin aikana suoritetaan. Batch size määrittää sen kuinka monen datapisteen jälkeen mallin parametreja päivitetään. Lisäksi malliin lisätään stop-funktio estämään ylisovitusta.

# Tulokset

Mallin selitysaste = 0.610
Mean Squared error = 0.008
Mean Absolute error = 0.07 

![mallin ennustavuus](/kuvat/regressio.png)

Mallin vertailukohtana syväoppimisregressiossa voidaan käyttää esimerkiksi keskiarvoa tai mediaania, joka olisi yksinkertaisin ennuste datasta tässä tapauksessa baseline MSE 0.115, joka on korkeampi kuin mallin tuottama. Ainakin malli ennustaa siis paremmin todennäköisyyttä tulla valituksi yliopistoon kuin mallia ei olisi ollenkaan.

# Hyperparametrit

Parhaimman mahdollisen mallin löytämiseksi mallia koottiin pala palalta lisäämällä piilotettuja kerroksia ja säätämällä hyperparametreja: batch sizea, epochs ja learning ratea.

Ensimmäisessä mallissa oli yksi piilotettukerros, jossa oli 8 neuronia. Malli ajettiin batch sizella 8, 16, 32.


![malli yksi](/kuvat/yksi_hidden_layer_8_nodea_batch_size_8_16_32_LR_001_epochs_300.png)

Sama malli, mutta batch sizea nostettiin 128, 256, 512.

![malli kaksi](/kuvat/yksi_hidden_layer_8_nodea_batch_size_128_256_512_LR_001_epochs_300.png)


Toisessa mallissa oli edelleen yksi piilotettukerros, jossa oli 16 neuronia. Batch sizella 32, 64, 128

![malli kolme](/kuvat/yksi_hidden_layer_16_nodea_batch_size_32_64_128_LR_001_epochs_300.png)

Kolmanteen malliin asetettiin kaksi piilotettua kerrosta 8, 16 neuronilla. Batch sizella 32, 64, 128

![malli nelja](/kuvat/kaksi_hidden_layer_8_ja_16_nodea_batch_size_32_64_128_LR_001_epochs_300.png)


Näytti siltä, että malli meni tähän ongelmaan nähden todella nopeasti liian monimutkaiseksi ja mallit näyttivät ylioppivan datan, lisäksi mallien selitysaste laski. Parhaimmaksi malliksi osoittaitui yhden piilotetun kerroksen malli, jossa on 8 neuronia batch sizella 128.

![lopullinen malli](/kuvat/lopullinen.png)
