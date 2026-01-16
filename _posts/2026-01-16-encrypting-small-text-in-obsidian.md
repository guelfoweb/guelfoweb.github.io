---
layout: post
title: "Encrypting small pieces of text in Obsidian, my way"
categories: [projects]
tags: [obsidian, templater, encryption, security]
---

I recently published a small project called **obsidian-text-lock**.

I want to say this immediately: I did not start this project to build “the next Obsidian plugin”. I wrote it to solve a very specific problem I have in my daily notes.

I do not need to encrypt my entire vault. I do not even need to encrypt whole notes. What I often need is much simpler. Inside my notes there are small portions of text that I would prefer not to leave in plain view. Things like temporary passwords, access tokens, or short private annotations mixed with normal content.

There are already plugins that can do this, and they work well. The problem is not quality. The problem, for me, is accumulation. Every plugin is another thing to install, configure, update, and trust over time. For something as simple as encrypting selected text, that felt excessive.

I already use **Templater** extensively in Obsidian. I use it for automation, note generation, and small scripting tasks. At some point I realized that everything I needed was already there. I could select text, run JavaScript, and access the *Web Crypto API*. So I asked myself a simple question: why add another plugin when the tool I already use can do the job?

That is how obsidian-text-lock was born.

While working on this project, I used ChatGPT as a support tool to better understand some Obsidian and Web Crypto API details, and to get feedback during code review.

Technically, **it is not really a plugin**. It is just two Templater templates. One encrypts the selected text, the other decrypts it. There is no background process, no interface, and no vault-wide behavior. You select text, run the template, and the selection is replaced. When you need the text back, you do the opposite.

When a piece of text is encrypted, the note shows a small lock marker and a short message. The encrypted data itself is stored inside an Obsidian comment, so it stays hidden in Preview mode but remains part of the Markdown file. This means the note stays readable, and the encrypted block does not visually pollute the content.

![Encrypted selection](https://github.com/guelfoweb/obsidian-text-lock/raw/main/screenshots/encrypted-selection.png)

From a security point of view, I deliberately kept things boring and standard. Encryption is done using `AES-256-GCM`, and the key is derived from a password using `PBKDF2` with `SHA-256`. There is no custom cryptography and no home-made tricks. Each encrypted block contains everything it needs to be decrypted later: `salt`, `IV` (nonce), and ciphertext. **The password is the only secret**. If you lose it, the data is gone. That is not a bug, it is the expected behavior.

There are also clear limitations, and I think it is important to be honest about them. 

- This works reliably only in *Source mode*, because of how Templater accesses text selections. 
- There is no key management, no recovery, and no protection against someone simply deleting the encrypted block.

This tool is meant for personal notes and low-risk scenarios, not for highly sensitive or regulated data.

I decided to publish *obsidian-text-lock* because it is small, transparent, and easy to understand. It does not try to replace existing plugins, and it does not aim to be feature-rich. It simply solves a narrow problem in a way that fits my workflow.

If you already use Templater and want a minimal way to encrypt small parts of your notes, this might be useful to you.

The project is available on GitHub:
[https://github.com/guelfoweb/obsidian-text-lock](https://github.com/guelfoweb/obsidian-text-lock)
