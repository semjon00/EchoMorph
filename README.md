EchoMorph v0

Features:
* Speaker encoder: allows zero-shorting speakers.
* History: gives better coherency for the output, allows not transmitting through the bottleneck some info that can
           be derived from the history.
* Randomask: allows to not train 8 different models with different sizes of bottleneck, but use the same model with
           configurable bottleneck size.
* Cris-cross: Propagates attention in different ways so signal can be propagated in different ways.
* Repeating blocks: Allows modifying quality

Dataset (obtained is a way that is absolutely not legally questionable, trust me bro guarantee):
* Talks in 200 languages, males
* Audiobooks in English, Estonian, Russian
* "Synthesized voices" in Japanese
* Solo music for 5 different musical instruments