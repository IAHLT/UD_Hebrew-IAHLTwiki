# Summary

Publicly available subset of the IAHLT UD Hebrew Treebank's Wikipedia section (https://www.iahlt.org/)

# Releases

Releases with pretrained models are provided as squashfs images for technical
reasons. To extract, install `squashfs-tools` and run e.g.:
```
unsquashfs UD_Hebrew-IAHLTwiki-r2.11.squashfs
```

The release will be placed in a folder `squashfs-root`.

# Introduction

The UD Hebrew-IAHLTWiki treebank consists of 5,000 contemporary Hebrew sentences representing a variety of texts originating from Wikipedia entries, compiled by the [Israeli Association of Human Language Technology](https://www.iahlt.org/). It includes various text domains, such as: biography, law, finance, health, places, events and miscellaneous. The schema for the UD Hebrew-IAHLT treebank, from which the publicly available UD Hebrew-IAHLTWiki subset is derived, is based on the conversion of the Hebrew Treebank (HTB) into the latest UD V2 and is checked against the Universal Dependencies validator as of UD release V2.10, in addition to a range of additional validations using the grewv tool.

The HTB version used in the project was initially converted automatically, then a subset of the converted data was manually validated and adopted as a gold standard for training the model for UD parsing used in Hebrew-IAHLT. The entire parsed data has been manually edited to correct parsing errors, and was automatically QA'ed to apply corrections following updates in the schema. 

# Acknowledgments

We would like to thank all the people who contributed to this corpus: Amir Zeldes, Hilla Merhav, Israel Landau, Netanel Dahan, Nick Howell, Noam Ordan, Omer Strass, Shira Wigderson, Yael Minerbi, Yifat Ben Moshe

## References

To cite this dataset please refer to the following paper:

Zeldes, Amir, Nick Howell, Noam Ordan and Yifat Ben Moshe (2022) [A Second Wave of UD Hebrew Treebanking and Cross-Domain Parsing](https://arxiv.org/abs/2210.07873). In: *Proceedings of EMNLP 2022*. Abu Dhabi, UAE.

```
@InProceedings{ZeldesHowellOrdanBenMoshe2022,
  author    = {Amir Zeldes and Nick Howell and Noam Ordan and Yifat Ben Moshe},
  booktitle = {Proceedings of {EMNLP} 2022},
  title     = {A SecondWave of UD Hebrew Treebanking and Cross-Domain Parsing},
  year      = {2022},
  address   = {Abu Dhabi, UAE},
}
```


# Changelog

* 2022-05-15 v2.10
  * Initial release in Universal Dependencies.


<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.10
License: CC BY-SA 4.0
Includes text: yes
Genre: wiki
Lemmas: manual native
UPOS: manual native
XPOS: not available
Features: manual native
Relations: manual native
Contributors: Zeldes, Amir; Algom, Avner; Ordan, Noam; Ben Moshe, Yifat; Wigderson, Shira
Contributing: here
Contact: amir.zeldes@georgetown.edu
===============================================================================
</pre>
