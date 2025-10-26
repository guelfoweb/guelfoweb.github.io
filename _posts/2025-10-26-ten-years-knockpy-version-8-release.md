---
layout: post
title: "Ten years of Knockpy: version 8 released"
categories: [projects]
tags: [knockpy, subdomains]
---

Knockpy is a small project I have been working on for about ten years. It started from curiosity and need, and over time it became a stable tool, used in many penetration testing distributions.
Through the years, it has proved useful in an important area of security and OSINT: subdomain reconnaissance, a key step for those who map the exposed assets of an organization.

The goal has always been to keep the tool simple, portable, and easy to adapt to different situations. I never wanted to make it something huge or very complex, but rather something that works well, clearly, and helps people who work with these tasks every day.

#### Recognitions that are good for the code

One of the things that makes me happiest, even after many years, is when a researcher writes to me privately to say thanks, or mentions Knockpy in a report or public post. Knowing that the tool really helped to find vulnerabilities or to achieve results in bug bounty programs is a quiet but deep satisfaction. Not because the credit is mine, but because the project, in its small way, has been useful to someone. And that, for me, gives meaning to the time spent keeping it alive.

## What has changed

With version 8, I decided to deeply review the internal architecture. This need came with time, from some limitations that appeared as technologies evolved, and also from the wish to make Knockpy easier to integrate into modern environments. Many parts of the original code had become hard to extend, and some choices made years ago no longer made sense today.

#### Asynchronous DNS engine

The first step was to clearly separate the different modules: DNS resolution first, then validation, and result saving. Each component is now isolated and can be used independently. This makes it possible, for example, to use the asynchronous DNS engine without running the full scanning pipeline, or to build a custom analysis with dynamic wordlists and specific parameters.

The DNS resolution part was completely rewritten using `asyncio`, to get better performance without losing stability. I tried to keep compatibility with existing tools and at the same time make the output cleaner, more consistent, and easier to use. Results are saved in JSON format, organized by domain, so they can be easily used in other analyses or automations.

#### HTTP response content in bytes and supported TLS protocol version

Alongside these structural changes, I added two new features designed to improve the analysis of active subdomains. The first shows the size in bytes of the HTTP response content; the second checks the TLS protocol, allowing the detection of supported versions and possible vulnerabilities. These data are also included in the JSON output, making them easy to extract and combine.

#### Optimized Python module

Since the previous version, Knockpy can be used both from the command line and as a Python module. This gives more flexibility for those who want to integrate it into their own scripts.

## AI helps, but human judgment remains essential

During the development of this version, I decided to experiment a bit. For some functions, especially the more structural ones, I used help from artificial intelligence tools. The experience was instructive. In some cases, I received good suggestions, useful for seeing the code from new angles or rethinking certain choices. In other cases, the code made by AI did not match the style and logic of the project.
The time spent reviewing, fixing, and adapting was still valuable. I learned that AI can be a good assistant, but it cannot replace the thinking and responsibility that every development choice needs. In any case, it was an interesting collaboration, and I think it could have a role again in the future if used carefully.

## Main features

### Simple resolution of a domain

```bash
$ knockpy -d guelfoweb.com

guelfoweb.com ['185.199.110.153', '185.199.108.153', '185.199.109.153', '185.199.111.153']
http   [301, 'https://guelfoweb.com/', 'GitHub.com', 162]
https  [200, None, 'GitHub.com', 6681]
cert   [True, '2025-12-25', 'guelfoweb.com', ['TLS 1.2', 'TLS 1.3']]
------------------------------------------------------------
1 domains in 00:00:00
```

With the `-d` parameter you set the domain. The first line shows the domain name and the list of IP addresses that resolve it.

The answers for the `http` and `https` protocols are shown in this order:

* `status_code`, `redirect`, `webserver`, `content_byte`

`cert` gives information about the certificate and the TLS protocol. It returns `True` if no problems are found; otherwise it returns `False`. The items shown in the list are:

* `True`, `expiration_date`, `subjectAltName`, `TLS_supported`

Ecco la traduzione in inglese (livello A2–B1):

### Subdomain reconnaissance

```bash
$ knockpy -d guelfoweb.com --recon
Reconnaissance...
- VirusTotal: ✔️
- Shodan:     ✔️
Scanned 3/3 domains...

www.guelfoweb.com ['185.199.109.153', '185.199.110.153', '185.199.111.153', '185.199.108.153']
http   [301, 'https://guelfoweb.com/', 'GitHub.com', 162]
https  [301, 'https://guelfoweb.com/', 'GitHub.com', 162]
cert   [True, '2025-12-25', 'guelfoweb.com', ['TLS 1.2', 'TLS 1.3']]
------------------------------------------------------------
guelfoweb.com ['185.199.109.153', '185.199.110.153', '185.199.108.153', '185.199.111.153']
http   [301, 'https://guelfoweb.com/', 'GitHub.com', 162]
https  [200, None, 'GitHub.com', 6681]
cert   [True, '2025-12-25', 'guelfoweb.com', ['TLS 1.2', 'TLS 1.3']]
------------------------------------------------------------
2 domains in 00:00:07
```

Adding the `--recon` option runs an online scan for subdomains. For each subdomain, the tool resolves it and shows the results in the order they are found.

#### API Key

For deeper checks, it is strongly recommended to set up `VirusTotal` and `Shodan` APIs. You can set the environment variables in two ways:

##### 1. Using a file named `.env` (recommended):

```bash
API_KEY_VIRUSTOTAL=your-virustotal-api-key
API_KEY_SHODAN=your-shodan-api-key
```

##### 2. Using a Unix/Linux shell command:

```bash
export API_KEY_VIRUSTOTAL=your-virustotal-api-key
export API_KEY_SHODAN=your-shodan-api-key
```

#### Bruteforcing

This is not a real brute-force attack, but a wordlist-based attack. To enable it, add the `--bruteforce` (or `--brute`) option. Knockpy will load the default list automatically.

```bash
knockpy -d guelfoweb.com --recon --brute
```

##### Wordlist

If you want to use your own wordlist, give its path with `--wordlist` followed by the file path.

```bash
knockpy -d guelfoweb.com --recon --brute --wordlist path/to/wordlist.txt
```

### Wildcard test

Testing for wildcard DNS is important before scanning. It avoids invalid results because the server could answer the same way for every subdomain. Run the test like this:

```bash
knockpy -d guelfoweb.com --wildcard
```

If the test is positive (wildcard is enabled), you do not need to continue the scan.

### Output of results

Each scan is saved automatically to a file named `domain.com_YYYY_MM_DD_HH_mm_ss.json`. In the example above, the file was saved as `guelfoweb.com_2025_10_25_21_46_40.json`.

#### Specific directory

To save the file in a specific folder, use `--save` and give the folder path. For example:

```bash
knockpy -d guelfoweb.com --recon --save path/to/results
```

#### JSON structure

Results are saved in JSON format, grouped by domain. This makes it easy to use them in other analyses or automations. The file `guelfoweb.com_2025_10_25_21_46_40.json` has this structure:

```
[
  {
    "domain": "www.guelfoweb.com",
    "ip": [
      "185.199.109.153",
      "185.199.110.153",
      "185.199.111.153",
      "185.199.108.153"
    ],
    "http": [
      301,
      "https://guelfoweb.com/",
      "GitHub.com",
      162
    ],
    "https": [
      301,
      "https://guelfoweb.com/",
      "GitHub.com",
      162
    ],
    "cert": [
      true,
      "2025-12-25",
      "guelfoweb.com",
      [
        "TLS 1.2",
        "TLS 1.3"
      ]
    ]
  },
  {
    "domain": "guelfoweb.com",
    "ip": [
      "185.199.109.153",
      "185.199.110.153",
      "185.199.108.153",
      "185.199.111.153"
    ],
    "http": [
      301,
      "https://guelfoweb.com/",
      "GitHub.com",
      162
    ],
    "https": [
      200,
      null,
      "GitHub.com",
      6681
    ],
    "cert": [
      true,
      "2025-12-25",
      "guelfoweb.com",
      [
        "TLS 1.2",
        "TLS 1.3"
      ]
    ]
  }
]
```

#### View a report

To show the results in a more readable way, use `--report` with the JSON file path. In the example:

```bash
knockpy --report guelfoweb.com_2025_10_25_21_46_40.json
```

## Python API

If installed, Knockpy can be imported as a Python module, which makes it easy to use.

```python
from knock import KNOCKPY

domain = 'example.com'

results = KNOCKPY(
    domain,
    dns="8.8.8.8",
    useragent="Mozilla/5.0",
    timeout=2,
    threads=10,
    recon=True,
    bruteforce=True,
    wordlist=None,
    silent=False
)

for entry in results:
    print(entry['domain'], entry['ip'], entry['http'], entry['cert'])
```

For production, set `silent=True` to avoid printing scan details during runs.

**Project link:** [https://github.com/guelfoweb/knock](https://github.com/guelfoweb/knock)
