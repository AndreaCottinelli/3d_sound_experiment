# 3d_sound_experiment
If you are here you, you know what we wanna do.


In breve come deve funzionare il codice.
L'idea di base è che una persona non vedente possa manipolare un modello stampato 3d di un cratere lunare così da percepirne con il tatto la distribuziona spaziale del modello. Lanciando lo script da una macchina dotata di webcam orientata indicativamente sul modello (dall'alto) è possibile associare delle mappe del modello in scale di grigi o con macchie di grigi che mappano diverse qualità fisiche del modello che possono essere espresse su una scala (età della superficie, densità di una sostanza chimica, albedo...) oppure a bin (regioni specifiche, composizione chimica principale...)

Quindi grazie alla sonificazione del modello 3d, possiamo codificare potenzialmente infinite mappe, su un solo modello fisico.


MODULO 1
Usa le fotografie delle scacchiere per estrarre i parametri di distorsione della telecamera, va lanciato ogni volta che si usa una nuova telecamera.

MODULO 2
- carica la mappa in scala di grigi (selezionabile)
- accede alla webcam
- riconosce la posizione della punta degli indici
- riconosce la posizione del modello 3d su cui stanno gli indici (magari tramite qr codes agli angoli??)
- proietta la mappa sul modello o la posizione degli indici sulla mappa
- in base al colore della mappa nella posizione delgi inici il codice produce due suoni nel range selezionato e li emette negli speaker corrispondenti ( indice sx nello speaker sx, e viceversa)

NOTE
- per ora mappiamo il bianco/nero ovver 0-1 tra due frequenze limite, poi svilupperemo diversi suoi e cose.
- sarebbe bellissimo fosse tutto pronto per il 9 giugno

