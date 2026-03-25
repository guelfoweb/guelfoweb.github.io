---
layout: post
title: "AGENTS.md su progetti complessi"
categories: [ai, agents]
tags: [agents.md, security]
---

# AGENTS.md su progetti complessi

Questa non è una guida. Voglio solo condividere la mia esperienza, soprattutto su progetti complessi dove uso l’AI in modo continuo e dove iniziano a pesare gli aspetti progettuali e di sicurezza.

Non ha molto senso cercare di scrivere un `AGENTS.md` "perfetto" fin dall'inizio. Non funziona così. Più in generale, quando lavoro a qualcosa di strutturato lo affronto come in edilizia: si parte da un progetto di massima, poi si entra nei dettagli e inevitabilmente qualcosa cambia in corso d'opera. Non perché il piano iniziale fosse sbagliato, ma perché alcune cose diventano chiare solo mentre si costruisce.

Leggendo in giro leggo spesso che `AGENTS.md` viene presentato come la soluzione per rendere gli agenti più affidabili. In parte è vero, ma il problema non è avere il file, è **come lo si scrive**.


## Dove ho sbagliato all'inizio

All'inizio ho fatto quello che fanno in molti: ho trattato `AGENTS.md` come documentazione. Ho scritto file lunghi, pieni di contesto, architettura, regole generiche, note varie.

Il risultato era poco utile. L'AI ignorava parti del file o si comportava comunque in modo incoerente. Più aggiungevo contenuto, meno sembrava avere effetto.

Col tempo ho iniziato a vederla in modo diverso. Non come un README per l’AI, che resta utile ma ha un altro scopo, ma come un insieme di vincoli operativi. Qualcosa che serve a limitare il comportamento del modello, non a spiegargli tutto.

## Lunghezza e contesto

Qui secondo me c'è un punto spesso sottovalutato. La lunghezza non è solo una questione di leggibilità ma è un limite tecnico.

Tutto quello che mettiamo in `AGENTS.md` finisce nel contesto del modello, e il contesto è limitato.

I modelli più recenti lavorano con contesti molto ampi, anche nell’ordine delle centinaia di migliaia di token o più, ma comunque finiti e condivisi tra prompt, codice e output.

Ogni riga in più compete con codice e istruzioni, quindi a un certo punto perde peso o viene ignorata. Per questo non mi convince molto l'idea di file troppo completi.


## Il modo in cui lo sto usando adesso

Quello che faccio ora è più semplice, ma molto più iterativo. C'è sempre un confronto continuo con il modello prima di arrivare al file.

Uso l'AI per ragionare sul progetto e sugli aspetti di sicurezza, per chiarire i punti critici e mettere a fuoco le scelte. Solo dopo chiedo di estrarre regole operative, poche, brevi e verificabili.

Poi faccio una revisione manuale, tolgo duplicati e frasi vaghe, e spesso torno di nuovo sul modello per riformulare o migliorare alcune regole. Solo alla fine costruisco `AGENTS.md`.

Il risultato è più piccolo, ma mi sembra più efficace. In particolare, quello che sembra fare davvero la differenza è la specificità. Regole generiche non aiutano, mentre quelle concrete fanno la differenza.


## Un approccio in evoluzione

Come detto all'inizio, `AGENTS.md` non è qualcosa che si scrive una volta e basta. Un progetto è un cantiere, cambia continuamente, e il file cambia insieme a lui.

Anche qui il confronto con il modello continua. Quando emergono nuovi problemi o voglio introdurre modifiche, torno a discuterne con l'AI, valuto alternative e poi aggiorno le regole. A volte si aggiungono vincoli, altre volte si semplifica o si rimuove ciò che non serve più.

Non credo esista ancora un metodo stabile o valido per tutti. Gli LLM non sono deterministici e anche il modo in cui interpretano queste regole può cambiare.

Per ora quello che mi sembra funzionare è abbastanza semplice: file corto, regole concrete, e niente delega completa al modello.

Il resto, almeno per me, è ancora sperimentazione.
