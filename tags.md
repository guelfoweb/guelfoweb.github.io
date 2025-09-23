---
layout: page
title: "Tags"
permalink: /tags/
---

<h2>Tag index</h2>
<ul>
{% for tag in site.tags %}
  {% capture name %}{{ tag | first }}{% endcapture %}
  <li><a href="#{{ name | slugify }}">{{ name }}</a></li>
{% endfor %}
</ul>

<hr>

{% for tag in site.tags %}
  {% capture name %}{{ tag | first }}{% endcapture %}
  <h2 id="{{ name | slugify }}">{{ name }}</h2>
  <ul>
    {% for post in site.tags[name] %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
