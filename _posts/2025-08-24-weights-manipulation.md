---
layout: post
title: "Weights manipulation"
categories: [ai, projects]
tags: [weights, abliteration]
---

Contenuto del post qui...


Some time ago I asked myself: do we really need many days of calculation and powerful GPUs to understand how an open weights language model manages its safety mechanisms? More important, is there a fast and reversible way that does not need the creation of abliterated models to make the model more obedient to specific requests?

From that question I started a research that was not easy. Making the code work with different models took time, because of many adjustments to fix library problems and memory limits.

After I got a working version (not completely stable), I tried a different approach: change in real time the embedding weights of specific tokens, reducing step by step the ones linked to refusal (sorry, cannot, dangerous) and increasing the ones linked to compliance (sure, help, explain).

With some models this worked well: small changes to refusal tokens slowly weakened the safety mechanisms, with the changes applied directly in RAM while the model was running. This way, the language style of the original model is kept, because the weights on disk stay the same, and the changes are fully reversible by reloading the model. This method takes minutes of tests instead of the many hours needed for abliteration.

But with some newer models the challenge is different. It is not enough to change only the embeddings, because the safety strategies are deep in the architecture, making the system much more resistant.

The study showed an important change in architecture: older models often put safety in the token embeddings, while modern ones spread it across the full neural network.

I documented on GitHub a real use case and the steps I used to change the weights of an older open model, with more details about both the advantages and the limits.

Link: [weights-manipulation](https://github.com/guelfoweb/weights-manipulation)
