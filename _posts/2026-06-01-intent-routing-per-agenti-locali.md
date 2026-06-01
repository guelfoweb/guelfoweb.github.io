---
layout: post
title: "Intent Routing per agenti locali"
categories: [ai, ita]
tags: [tool-call, intent, cli]
---

Quando si progetta una CLI agentica per modelli piccoli si tende a pensare che più strumenti significhino più capacità. È un po' come mettere una persona davanti a una console piena di pulsanti e aspettarsi che trovi immediatamente quello giusto. In teoria può farlo. In pratica aumenta solo il numero di scelte possibili e quindi la probabilità di errore.

Con le `tool-call` succede qualcosa di molto simile. La tentazione è esporre tutto al modello: filesystem, shell, web, editing, vision, audio. 

In teoria il modello dovrebbe scegliere lo strumento corretto, ma nella pratica, soprattutto con modelli piccoli, succede spesso il contrario.

Un prompt come:

```text
show me how grep works
```

non dovrebbe eseguire `grep`: l'utente sta chiedendo una spiegazione. Ma un runtime troppo permissivo può interpretarlo come una richiesta operativa e aprire inutilmente l'accesso alla shell.

È un po' come chiedere a un meccanico come funziona un motore e vederlo aprire il cofano per iniziare a smontare la tua auto. La domanda riguarda una spiegazione, non un intervento.

Allo stesso modo:

```text
tell me about file systems
```

non significa "*ispeziona la directory corrente*". È una domanda concettuale.

Oppure:

```text
tell me about malware analysis, C2 and IoC
```

non significa "*analizza i file presenti nella workdir come malware sample*".

In questi esempi il sistema non ha sbagliato strumento. Ha sbagliato intenzione.

All'inizio pensavo che il problema fosse solo il modello. In realtà, **una parte importante del problema è il runtime**.

## Introdurre un livello di routing prima della tool-call

La soluzione richiede un compromesso: introdurre una piccola componente deterministica prima di lasciare spazio al modello aggiungendo un livello di routing prima della tool-call.

Il flusso non è più:

```text
prompt -> modello -> tutti i tool disponibili
```

ma qualcosa di più controllato:

```text
prompt -> intent routing -> eventuale intent gate -> subset minimo di tool -> modello
```

Il primo passaggio classifica la richiesta. Il runtime prova a capire se l'utente sta facendo una domanda generale, oppure sta chiedendo accesso al filesystem, una ricerca web, una modifica file, un'analisi immagine/audio o un'operazione più rischiosa.

Se l'intento è chiaro, il sistema espone solo i tool necessari.

Per esempio:

```text
read README.md
```

porta solo agli strumenti `filesystem`.

```text
search online for information about Dante Alighieri
```

porta solo agli strumenti `web`.

```text
compare image1.png and image2.jpg
```

porta solo al percorso `vision`.

### Gestione dei casi ambigui

Nei casi ambigui entra in gioco un secondo livello: l'**intent gate**.

È una piccola richiesta al modello, ma molto vincolata. Non gli viene chiesto di risolvere il task, viene chiesto solo se abbia senso procedere con tool locali oppure trattare il prompt come conversazione.

Esempio concettuale:

User prompt:
```text
tell me about malware analysis
```

Question to model:
```text
Should local filesystem/shell tools be used to perform malware analysis for this user request?

Answer only YES or NO.
```
Se la risposta è **NO**, il runtime **non espone i tool** e risponde normalmente.

Se la risposta è **YES**, **espone solo il subset** coerente con quell'intento.

Questa distinzione risolve diversi problemi pratici:

- Domande concettuali che prima attivavano shell o filesystem restano conversazione.
- Richieste operative reali continuano a usare i tool.
- I casi ambigui non vengono decisi da una regex rigida, ma da un controllo semantico molto stretto.
- Il modello vede meno strumenti, quindi ha meno possibilità di scegliere male.
- Si riducono loop inutili, consumo di token e comportamenti laterali.

La parte importante è che il gate non rende l'agente deterministico nel senso classico. Non decide il contenuto della risposta, decide solo se sia appropriato aprire una certa classe di capacità operative. Per un piccolo modello locale questa differenza è enorme.

In pratica, il modello non viene lasciato davanti a una cassetta degli attrezzi completa a ogni turno, ma gli viene dato solo quello che serve per la richiesta corrente.

Se a un muratore viene chiesto di misurare un muro, per quel compito bastano un metro, una matita e un foglio di carta. Mettergli davanti anche trapano, betoniera, martello demolitore e sega circolare non lo aiuta a lavorare meglio. Gli offre semplicemente più modi per fare qualcosa che non gli è stato chiesto.

Meno strumenti significa meno ambiguità, dunque meno errori, meno loop e meno token spesi per decidere cosa fare.

Uno dei modi migliori per rendere un agente più affidabile non è aggiungere capacità ma togliere quelle che non servono in quel momento.

