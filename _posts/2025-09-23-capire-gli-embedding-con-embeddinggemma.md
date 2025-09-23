---
layout: post
title: "Capire gli embedding con EmbeddingGemma"
categories: [ai, ita]
tags: [embedding, semantic, classification, clustering, rag]
---

Si parla molto di LLM, i cosiddetti _Large Language Models_ come ChatGPT, Gemini o Llama, modelli che sanno scrivere testi, rispondere a domande, riassumere documenti. Insomma addestrati per generare linguaggio.

Accanto a questa famiglia esiste un altro tipo di modello, meno conosciuto ma non per questo meno importante. Sono **i modelli di embedding**. A differenza degli LLM, questi non producono frasi, il loro scopo è quello di prendere un testo e trasformarlo in una sequenza di numeri, un vettore che ne rappresenta il significato.

Anche un LLM, per funzionare, utilizza internamente un sistema di embedding. Ogni parola, ogni pezzo di parola, viene trasformato in numeri prima di poter essere elaborato. La differenza è che negli LLM questo passaggio rimane nascosto, serve solo come base per arrivare alla generazione del linguaggio. Nei modelli di embedding, invece, questa trasformazione è l'obiettivo stesso.

Per intenderci, se chiediamo ad un LLM “***cos’è una firma digitale?***” ci risponderà con una spiegazione articolata. Un modello di embedding, alla stessa domanda, non scrive nessuna risposta testuale, restituisce invece un insieme di numeri che rappresentano quella frase nello **spazio semantico**, una sorta di mappa in cui ogni frase trova una posizione in base al suo significato.

Facciamo un esempio per chiarire meglio il concetto. Chiediamo al modello di embedding "*Cos'è una firma digitale?*". Non gli forniamo nessun documento da confrontare, vogliamo solo vedere cosa produce in uscita.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

query = "Cos'è una firma digitale?"

# Query embedding
query_emb = model.encode(query, normalize_embeddings=True)

print("Embedding size:", query_emb.shape)
print("Values:", query_emb)
```

Il risultato sarà il seguente:

```
Embedding size: (768,)
Values: [-1.29152596e-01 -2.52732392e-02 -1.64837558e-02  4.21979874e-02
  1.01330150e-02  4.22639251e-02 -9.70820710e-03  3.03274114e-02
 -2.69848965e-02  3.98598611e-03 -3.41513008e-02  2.13023946e-02
 -1.75719075e-02  7.05240667e-02  1.03565618e-01 -4.28020488e-03
  .......
  .......
 -1.63111847e-03  4.68430854e-02 -5.01399534e-03  1.51540190e-02
 -5.29640205e-02 -1.45818135e-02 -3.43373977e-02  2.02104747e-02
 -1.22715998e-02  6.40184134e-02 -4.49002022e-03  3.19903553e-03
 -4.00503315e-02 -8.90033245e-02  1.74124353e-02  1.15116490e-02]
```

Questi numeri costituiscono un **vettore rappresentativo** dell'intera frase. Nel caso di **EmbeddingGemma** (il modello di embedding che stiamo utilizzando) il vettore ha sempre una **dimensione fissa** di 768 valori, indipendentemente dalla lunghezza del testo. Possiamo pensare questi numeri come coordinate che permettono di confrontare una domanda (query) con altri testi e stabilire, tramite algoritmi che misurano la similarità o la distanza euclidea, se due frasi esprimono concetti simili oppure trattano argomenti molto diversi.

In generale possiamo dire che dati due o più vettori numerici, più le loro coordinate sono vicine, più le frasi che rappresentano condividono lo stesso significato; al contrario, se i vettori risultano distanti, significa che i testi corrispondenti parlano di argomenti molto diversi.

Per rendere il concetto più semplice, possiamo immaginare che ogni frase sia come una città su una mappa. L'embedding è come la **coppia di coordinate** (latitudine e longitudine) che ci dice dove si trova quella città. Se due città sono vicine, vuol dire che hanno molto in comune (regione, clima, cultura, economia, storia, lingua...), se invece sono lontane, vuol dire che appartengono a contesti diversi.

Se ora forniamo un documento dove dice che _“La firma digitale è un sistema informatico che assicura autenticità e integrità di un documento elettronico.”_, i due embedding, quello della domanda e quello del documento, finiranno in punti vicini della mappa semantica. 

Se invece confrontiamo la stessa domanda con un testo che parla del Colosseo, ad esempio *"Il Colosseo è un antico anfiteatro romano situato nel centro di Roma."*, i due punti saranno molto lontani.

Vediamolo con un esempio.

#### Codice_1
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("./embeddinggemma-300m")

# Documenti da confrontare
documents = [
    "La firma digitale è un sistema informatico che assicura autenticità e integrità di un documento elettronico.",
    "Il Colosseo è un antico anfiteatro romano situato nel centro di Roma."
]

# Query dell'utente
query = "Cos'è una firma digitale?"

# Calcolo degli embedding
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Calcolo la similarità coseno tra la query e i documenti
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)

# Trovo il documento più simile
best_idx = cosine_scores.argmax()
print("Documento più rilevante:", documents[best_idx])
print("Punteggio di similarità:", cosine_scores[0][best_idx].item())

print()

# Stampa di tutti i punteggi per confronto
for doc, score in zip(documents, cosine_scores[0]):
    print(f"{doc} → {score.item():.3f}")
```

Il risultato sarà il seguente:

```
Documento più rilevante: La firma digitale è un sistema informatico che assicura autenticità e integrità di un documento elettronico.
Punteggio di similarità: 0.7245991230010986

La firma digitale è un sistema informatico che assicura autenticità e integrità di un documento elettronico. → 0.725
Il Colosseo è un antico anfiteatro romano situato nel centro di Roma. → 0.244
```

In questo esempio il modello ha restituito come più rilevante la frase sulla firma digitale, con un punteggio di similarità (0.72) nettamente superiore rispetto a quella sul Colosseo (0.24). 

In realtà questo risultato non dovrebbe sorprenderci. La query `Cos’è una firma digitale?` e il documento `La firma digitale è un sistema informatico che assicura autenticità e integrità di un documento elettronico.` condividono esplicitamente le stesse parole chiave, in particolare l’espressione `firma digitale`. Questo facilita il compito del modello che può appoggiarsi anche alla corrispondenza lessicale. 

**Proviamo quindi con un esempio più difficile**

Per rendere il test più interessante proviamo con un esempio più difficile, con documenti che non contengano le stesse parole della query, in modo da verificare la capacità del modello di cogliere davvero la similarità semantica e non solo la somiglianza superficiale delle stringhe. 

Modifichiamo il primo documento in **Codice_1**.

```python
# Documenti da confrontare
documents = [
    "È un sistema che permette di verificare l’identità dell’autore di un file e di controllare che il contenuto non sia stato modificato.",
    "Il Colosseo è un antico anfiteatro romano situato nel centro di Roma."
]

# Query dell'utente
query = "Cos'è una firma digitale?"
```

Risultato:

```
Documento più rilevante: È un sistema che permette di verificare l’identità dell’autore di un file e di controllare che il contenuto non sia stato modificato.
Punteggio di similarità: 0.3943994641304016

È un sistema che permette di verificare l’identità dell’autore di un file e di controllare che il contenuto non sia stato modificato. → 0.394
Il Colosseo è un antico anfiteatro romano situato nel centro di Roma. → 0.244
```

La query è rimasta sempre la stessa: `Cos’è una firma digitale?`, ma il primo documento non contiene più la stessa espressione. Al posto di `firma digitale` viene usata una descrizione del concetto, parlando di un sistema che consente di verificare l’identità dell'autore di un file e di controllare che non sia stato modificato. 

`È un sistema che permette di verificare l’identità dell’autore di un file e di controllare che il contenuto non sia stato modificato.`

Nonostante questa differenza lessicale, il modello è riuscito a capire che quella descrizione si riferiva allo stesso concetto e l'ha collegata correttamente alla domanda. Il risultato è stato un punteggio più alto rispetto a un documento del tutto diverso, come quello sul Colosseo. In altre parole, non si è fermato alle singole parole, ma ha saputo cogliere il senso complessivo della frase.

## Come avviene il confronto con cosine similarity?

Supponiamo di avere la nostra domanda (query)  e le due frasi (doc1 e doc2) trasformati, per semplicità, in embedding a 5 dimensioni invece che 768.

```
query = "Cos’è una firma digitale?" 
E(query) = [0.010,  0.042, -0.009,  0.030, -0.015]

doc1 = "La firma digitale garantisce autenticità dei documenti"
E(doc1) = [0.012,  0.039, -0.011,  0.028, -0.017]

doc2 = "Il Colosseo si trova a Roma"
E(doc2) = [-0.045,  0.002,  0.037, -0.020,  0.041]
```

Indichiamo per comodità E(query) = `A`, E(doc1) = `B` e E(doc2) = `C`

Il confronto tramite **[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)** è molto semplice dal punto di vista matematico (e [questo video](https://www.youtube.com/watch?v=e9U0QAFbfLI) lo spiega abbastanza bene) in quanto si basa sul prodotto scalare e sulla norma dei vettori.

$$
\mathrm{sim}_{\cos}(A,B)=\frac{A\cdot B}{\|A\|\;\|B\|}
$$
ovvero:
$$
\mathrm{sim}_{\cos}(A,B)=
\frac{\sum_{i=1}^{n} A_i B_i}{
\sqrt{\sum_{i=1}^{n} A_i^{2}}\;\sqrt{\sum_{i=1}^{n} B_i^{2}}
}
$$

risparmiandoci i calcoli, visto che Claude Sonnet è bravo e veloce a fare i conti, otteniamo:

- **Cosine similarity (A,B) ≈**`1.028` (molto simili, coseno vicino a 1, quasi identici)
- **Cosine similarity (A,C) ≈**`-0.536` (frasi molto diverse, quasi in direzione opposta)

La cosine similarity guarda l’**angolo** tra le due frecce (vettori) nello spazio. Se l’angolo è piccolo, i vettori sono quasi paralleli, dunque sono frasi con lo stesso significato.

## Come fa EmbeddingGemma a lavorare con 100 lingue?

EmbeddingGemma **non ha cento vocabolari diversi** al suo interno, uno per ciascuna delle lingue su cui è stato addestrato. Usa invece un unico tokenizer multilingue, basato su subword, cioè pezzi di parola che vengono combinati per rappresentare testi in lingue diverse. 

Verifichiamo la parola italiana `digitale` e poi il termine inglese `digital`, per capire se vengono rappresentati come token unici o spezzati in sub-token e, soprattutto, per osservare come vengono convertiti dal modello.
#### Codice_2
```python
from transformers import AutoTokenizer

# Tokenizer EmbeddingGemma
model_id = "google/embeddinggemma-300m"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Word to test
word = "digitale"

# Get ID
ids = tokenizer.encode(word, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(ids)

print("Word:", word)
print("Token IDs:", ids)
print("Tokens:", tokens)

# Check if the word matches a unique token
if len(tokens) == 1:
    print("unique token")
else:
    print("sub-token")

```

Risposta per `digitale` in italiano
```bash
Word: digitale
Token IDs: [29345, 1203]
Tokens: ['digit', 'ale']
sub-token
```

Risposta per `digital` in inglese
```bash
Word: digital
Token IDs: [36661]
Tokens: ['digital']
unique token
```

Dai risultati possiamo osservare che `digital` in inglese è talmente frequente da essere un **token unico,** mentre l'italiano `digitale` viene spezzato (sub-token) in due pezzi: `digit` e `ale`. Nonostante la differenza, entrambi contengono il frammento `digit` che riduce la distanza tra le due rappresentazioni, ma è grazie all'addestramento multilingue che il modello impara ad allineare i significati e a considerare i due termini semanticamente vicini.

Il modello, dunque, non ha un token per ogni parola di ogni lingua, sarebbe impraticabile avere un vocabolario separato per 100 lingue. **EmbeddingGemma** lavora con un **set di mattoncini linguistici** (sub-token) che possono essere combinati per ricostruire le parole delle lingue su cui è stato addestrato.

Ad esempio `digitalization` diventa:
```
Word: digitalization
Token IDs: [36661, 1854]
Tokens: ['digital', 'ization']
sub-token
```

Il termine viene spezzato in due parti:

- `digital`: già presente come token unico e molto frequente (ID 36661).
- `ization`: suffisso comune in inglese, riusato in molte parole (_organization, realization, optimization…_).

## Cosa possiamo fare con EmbeddingGemma?

Come abbiamo già detto, EmbeddingGemma è stato ottimizzato esclusivamente per catturare la similarità semantica. Non scrive frasi, ma crea rappresentazioni numeriche che permettono di fare **ricerca semantica**, **classificazione**, **clustering** o di supportare applicazioni di **Retrieval-Augmented Generation** (RAG).

Vediamoli in ordine uno per uno.
### Ricerca semantica

Abbiamo visto che la query e i documenti vengono trasformati in vettori e poi confrontati tramite algoritmi di similarità. Prendiamo ad esempio un mini dataset in ambito cybersecurity e poniamo alcune domande utilizzando lo script di **Codice_1**.

**Dataset di documenti (frasi per il nostro esempio)**

1. `Il phishing è una tecnica fraudolenta che cerca di rubare credenziali fingendosi un ente affidabile.`
2. `Un ransomware è un malware che cripta i file di un computer e chiede un riscatto per sbloccarli.`
3. `Un attacco DDoS consiste nell'inviare un numero enorme di richieste a un server per renderlo inaccessibile.`
4. `L'autenticazione a due fattori (2FA) aumenta la sicurezza richiedendo un codice aggiuntivo oltre alla password.`
5. `Un firewall controlla il traffico di rete in ingresso e in uscita per proteggere i sistemi informatici.`

Se proviamo con le seguenti **query**, vediamo che il modello riesce a rispondere in modo corretto e senza difficoltà.

1. `Quale evento informatico blocca un server sommergendolo di richieste?`
2. `Quale inganno online induce una persona a consegnare informazioni private pensando di parlare con un ente affidabile?`
3. `Quale malware blocca l’uso dei documenti sul PC finché non viene versato denaro?`
4. `Quale procedura di accesso richiede la conferma tramite smartphone oltre all’inserimento tradizionale?`
5. `Quale tecnologia agisce come barriera tra un computer e Internet, impedendo intrusioni non autorizzate?`

Di seguito un esempio (n.5) di risposta:
```
Query: Quale tecnologia agisce come barriera tra un computer e Internet, impedendo intrusioni non autorizzate?
Documento più rilevante: Un firewall controlla il traffico di rete in ingresso e in uscita per proteggere i sistemi informatici.
Punteggio di similarità: 0.578133225440979
```

#### Migliorare il prompt per retrieval

Come suggerito in una discussione su [Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5/discussions/11) e come documentato nella guida di [Sentence-Transformers](https://sbert.net/examples/sentence_transformer/training/prompts/README.html), l'uso dell'istruzione `Represent this sentence for searching relevant...` prima della query può aiutare il modello a interpretare meglio il compito e, di conseguenza, migliorare le prestazioni nei task di retrieval.

La query (in *Codice_1*) verrà quindi preceduta dall'istruzione per migliorare la qualità degli embedding nelle attività di retrieval semantico.

```python
prompt = "Represent this sentence for searching relevant documents: "
query = "Cos'è una firma digitale?"
query_prompted = prompt + query
```

### Classificazione

Un modello di embedding non decide da solo se ad esempio un'email è phishing o no, si limita a trasformare il testo in numeri che ne rappresentano il significato. Quei numeri diventano la materia prima per un classificatore. In questo esempio verrà usato `LogisticRegression`, un semplice classificatore della libreria scikit-learn.

Immaginiamo di avere solo 12 email, metà legittime e metà di phishing. Useremo alcune di esse per insegnare al classificatore a riconoscere la differenza e le altre per metterlo alla prova. Alla fine confronteremo le risposte con quelle corrette e calcoliamo quante ne ha indovinate.
#### Codice_3
```python                                                       
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    # Phishing (1)
    "Aggiorna subito la tua password cliccando su questo link",
    "Hai vinto un premio! Inserisci i tuoi dati bancari per riceverlo",
    "Il tuo conto è stato bloccato, verifica immediatamente le tue credenziali",
    "Gentile cliente, la tua carta di credito è sospesa. Accedi qui per sbloccarla",
    "Riceverai un rimborso, basta compilare il modulo online con i tuoi dati",
    "Il tuo account verrà chiuso se non confermi subito l’accesso",

    # Legittime (0)
    "La riunione del team è fissata per domani alle 10",
    "Grazie per aver acquistato sul nostro sito, trovi la fattura in allegato",
    "Il corso di formazione inizierà la prossima settimana",
    "Ecco il verbale della riunione di ieri",
    "La consegna del tuo pacco è prevista per giovedì",
    "La biblioteca comunale resterà chiusa per lavori fino a fine mese"
]

labels = [1,1,1,1,1,1, 0,0,0,0,0,0]

# Carica EmbeddingGemma
model = SentenceTransformer("google/embeddinggemma-300m")

# Calcola gli embedding delle email
X = model.encode(emails, normalize_embeddings=True)
y = labels

# Split train/test
test_size = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Stampa numeri e percentuali
n_total = len(emails)
n_train = len(X_train)
n_test = len(X_test)

print(f"Totale esempi: {n_total}")
print(f"Training set: {n_train} esempi ({n_train/n_total:.0%})")
print(f"Test set: {n_test} esempi ({n_test/n_total:.0%})\n")

# Allena classificatore
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predizione
pred = clf.predict(X_test)

print("Predizioni:", pred.tolist())
print("Valori reali:", y_test)
print("Accuratezza:", accuracy_score(y_test, pred))
```

Con `train_test_split` il dataset viene diviso in due parti, esattamente al 50% visto che `test_size=0.5`:

1. **Training test**: usato per addestrare il classificatore, il modello impara da questa porzione.
2. **Test set**: messo da parte e usato solo per valutare, sono le email che userà in `.predict(X_test)`

Vediamo alcuni risultati cambiando i valori di `test_size`:

Risultati per `test_size=0.2`: indovinate **2 su 3** con una accuratezza del **66,7%**.
```
Totale esempi: 12
Training set: 9 esempi (75%)
Test set: 3 esempi (25%)

Predizioni: [1, 0, 1]
Valori reali: [0, 0, 1]
Accuratezza: 0.6666666666666666
```

Risultati per `test_size=0.5`: indovinate **5 su 6** con una accuratezza del **83,3%**.
```
Totale esempi: 12
Training set: 6 esempi (50%)
Test set: 6 esempi (50%)

Predizioni: [0, 0, 1, 0, 0, 1]
Valori reali: [0, 0, 1, 0, 1, 1]
Accuratezza: 0.8333333333333334
```

Risultati per `test_size=0.8`: indovinate **8 su 10** con una accuratezza del **80%**.
```
Totale esempi: 12
Training set: 2 esempi (17%)
Test set: 10 esempi (83%)

Predizioni: [0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
Valori reali: [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
Accuratezza: 0.8
```

I risultati cambiano molto a seconda di come dividiamo i dati tra training e test. Con il 75% dei dati usati per l’addestramento l’accuratezza è stata del 66,7%, con una divisione a metà è salita all’83,3%, mentre con appena 2 esempi usati per allenare il modello (e ben 10 per testarlo) si è comunque mantenuta intorno all’80%.

È bene ricordare che si tratta di un test fatto con pochissimi esempi, non possiamo aspettarci stabilità nei numeri. La cosa importante, però, è che la pipeline funziona. EmbeddingGemma riesce a trasformare il testo in numeri che catturano il significato e, a partire da questi, persino un classificatore semplicissimo come la Logistic Regression riesce a distinguere tra phishing ed email legittime. Con un numero maggiore di dati reali i risultati diventerebbero molto più solidi e affidabili.

È un pò come insegnare a un bambino a distinguere tra frutta e verdura: se gli mostriamo solo pochi esempi all'inizio farà confusione, ma man mano che gli facciamo vedere altri casi imparerà a riconoscerle sempre meglio.

### Clustering

Mentre nella **ricerca semantica** c’è sempre una query, nel **clustering**, invece, non c’è nessuna query. Lo scopo è quello di scoprire gruppi di testi simili. 

Prendiamo un insieme di testi, li trasformiamo tutti in embedding e lasciamo che un algoritmo di clustering (come `k-means`) scopra automaticamente i gruppi. L'idea è che testi simili finiranno nello stesso cluster, anche senza etichette. È un approccio esplorativo che serve a capire come si organizzano i dati da soli.

Il clustering è un ottimo metodo per scoprire strutture nascoste nei dati.

Proviamo a fare un esempio concreto con 6 frasi e applichiamo il `k-means` con due cluster. Ad occhio (sono pochi documenti) è facile distinguere che un cluster riguarda i **luoghi** e l'altro i **servizi digitali**, ma è interessante verificare come vengono suddivisi.

#### Codice_4
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Carica EmbeddingGemma
model = SentenceTransformer("google/embeddinggemma-300m")

# Dataset di frasi di esempio
sentences = [
    "Il Colosseo si trova a Roma",
    "La Torre Eiffel è a Parigi",
    "Il Vesuvio è un vulcano vicino a Napoli",
    "Lo SPID permette di accedere ai servizi online della Pubblica Amministrazione",
    "La Carta d’Identità Elettronica può essere usata per l’accesso ai servizi digitali",
    "Ho dimenticato la password del mio account SPID"
]

# Calcola gli embedding
embeddings = model.encode(sentences, normalize_embeddings=True)

# Applica k-means con 2 cluster (luoghi e servizi digitali)
num_clusters = 2
clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

# Stampa i risultati
clusters = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clusters[cluster_id].append(sentences[sentence_id])

for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}:")
    for sentence in cluster:
        print(" -", sentence)
```

Il raggruppamento per somiglianza ha prodotto i due gruppi di testi vicini nello spazio semantico, ma senza applicare etichette.

```
Cluster 1:
 - Il Colosseo si trova a Roma
 - La Torre Eiffel è a Parigi
 - Il Vesuvio è un vulcano vicino a Napoli

Cluster 2:
 - Lo SPID permette di accedere ai servizi online della Pubblica Amministrazione
 - La Carta d’Identità Elettronica può essere usata per l’accesso ai servizi digitali
 - Ho dimenticato la password del mio account SPID
```

Possiamo pensare al clustering con `k-means` come a una forma di **auto-classificazione**. È un pò come prendere una scatola piena di documenti e chiedere al modello di dividerli in pile in base a quello che gli sembra più simile, senza dirgli prima quali etichette usare. Magari una pila conterrà documenti che parlano di riunioni, un'altra quelli che parlano di conti bancari. Non è detto che i gruppi coincidano sempre con le categorie che avevamo in mente, per questo spesso rivelano strutture interessanti nei dati.

### RAG (Retrieval-Augmented Generation)

Il RAG è il punto d'incontro tra due mondi: da un lato gli **embedding**, che servono a cercare e recuperare i testi più rilevanti, e dall'altro gli **LLM**, che hanno la capacità di generare risposte articolate in linguaggio naturale. 

Se prima abbiamo visto la **ricerca semantica**, utile per trovare il documento più vicino a una query, la **classificazione**, dove insegniamo al modello a distinguere testi etichettati, e il **clustering**, che invece raggruppa automaticamente testi simili, con il **RAG** facciamo un passo in più. Qui gli embedding ci aiutano a recuperare le informazioni giuste e poi lasciamo che sia l'LLM a costruire una risposta finale chiara e leggibile per l'utente.

Vediamo un esempio concreto con `transformers` e `Gemma-2b-it` come LLM open source.
#### Codice_5
```python
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# 1. Embedding model per retrieval
embedder = SentenceTransformer("google/embeddinggemma-300m")

# Documenti
documents = [
    "Lo SPID è un sistema di autenticazione che consente ai cittadini italiani di accedere ai servizi online della Pubblica Amministrazione.",
    "La Carta d’Identità Elettronica (CIE) può essere usata per l’accesso ai servizi digitali.",
    "Il Colosseo è un antico anfiteatro romano situato a Roma."
]

# Query
query = "Qual è il sistema che permette di accedere ai servizi online della Pubblica Amministrazione?"

# 2. Retrieval
doc_embeddings = embedder.encode(documents, convert_to_tensor=True)
query_embedding = embedder.encode(query, convert_to_tensor=True)

cosine_scores = util.cos_sim(query_embedding, doc_embeddings)
best_idx = cosine_scores.argmax().item()
retrieved_doc = documents[best_idx]

print("Documento più rilevante:", retrieved_doc)

# 3. Passiamo la query + documento a un LLM
generator = pipeline("text-generation", model="google/gemma-2b-it")

prompt = f"Domanda: {query}\n\nContesto: {retrieved_doc}\n\nRisposta:"
output = generator(prompt, max_new_tokens=100)[0]["generated_text"]

print("\nRisposta generata:\n", output)
```

Vediamo il risultato ottenuto:

```
Documento più rilevante: Lo SPID è un sistema di autenticazione che consente ai cittadini italiani di accedere ai servizi online della Pubblica Amministrazione.

Risposta generata:
 Domanda: Qual è il sistema che permette di accedere ai servizi online della Pubblica Amministrazione?

Contesto: Lo SPID è un sistema di autenticazione che consente ai cittadini italiani di accedere ai servizi online della Pubblica Amministrazione.

Risposta: Il sistema SPID è il sistema di autenticazione per i servizi online della Pubblica Amministrazione.
```

In questo caso, EmbeddingGemma ha fatto bene il suo lavoro di **retrieval**, ma ha riportato **solo il documento più vicino**, quello sullo SPID. L'LLM ha quindi costruito la risposta basandosi su quel contesto, **ignorando la CIE** che era comunque presente negli altri documenti. Questo succede perchè nel nostro esempio abbiamo utilizzato solo tre frasi come documenti. Nei sistemi RAG reali non si passa solo **1 documento**, ma un piccolo set, potrebbero essere i primi 3 più rilevanti, così l’LLM ha più materiale per generare una risposta completa.

## Conclusione

Dal punto di vista pratico, EmbeddingGemma è molto leggero: ha circa 300 milioni di parametri, occupa meno di 200 MB di RAM e accetta input fino a 2048 token. L'output è un embedding di 768 dimensioni, che può essere ridotto nel caso si voglia ottimizzare memoria e velocità. 

Grazie alle sue dimensioni contenute, EmbeddingGemma può essere usato anche su dispositivi modesti, senza bisogno di GPU dedicate. Questo lo rende ideale per sperimentazioni locali, per applicazioni leggere o per scenari edge, dove non è pratico affidarsi al cloud. In altre parole, è un modello accessibile a chiunque voglia lavorare con gli embedding senza infrastrutture costose.

---
## Note su EmbeddingGemma

[**EmbeddingGemma**](https://deepmind.google/models/gemma/embeddinggemma/) è un modello di embedding sviluppato da Google DeepMind ed è open-source. Nasce sulla base dell’architettura Gemma 3, la stessa famiglia di modelli LLM rilasciata da Google, ma è stato adattato appositamente per fare embedding e non generazione. Prima di essere addestrato sugli embedding, i suoi pesi sono stati inizializzati con quelli di [*T5Gemma*](https://deepmind.google/models/gemma/t5gemma/), così da partire già con una solida comprensione linguistica (grammatica, semantica, multilinguismo) e specializzarsi poi nell'unico compito di rappresentare i testi come vettori numerici.

