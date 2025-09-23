---
layout: page
title: "All categories"
permalink: /categories/
---

<h2>Categories</h2>
<ul>
{% for category in site.categories %}
  {% capture name %}{{ category | first }}{% endcapture %}
  <li><a href="#{{ name | slugify }}">{{ name }}</a></li>
{% endfor %}
</ul>

<hr>

{% for category in site.categories %}
  {% capture name %}{{ category | first }}{% endcapture %}
  <h2 id="{{ name | slugify }}">{{ name }}</h2>
  <ul>
    {% for post in site.categories[name] %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
