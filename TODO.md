juste je crois que le repo s'est renommé wiki tout court. 

Aussi, la tolérance donnée par le pire scénraio avec truncation (Delta_N) est vraiment pourrie, il n'y a pas de meilleure tolérance?

L'algorithme et son appendix ne sont pas très bien structurés, il faut les garder rigoureux mais on ne voit pas bien quelles sont les étapes intéressantes.

Enfin, je veux que tu utilises optyx pour les tests (en gardant les fichiers python dans les md de content/photonic). Je veux que tu utilises pour ça ma version dans exp/ optyx-src (c'est la branche du repo git qui implémente l'agorithme).

Mais, je veux en revanche que tu réimplémentes l'algo de façon très minimaliste, pars de la PR15 du repo et change uniquement unroll_certificate, en raisant une error si la tolérance ne peut pas être atteinte (en spéciufiant si c'est la truncation OU carrément le nombre de steps trop grand).

pardon juste pour le prompt précédent, ne raise pas une erreur mais un warning et précise la tolérance permise par le max_n_steps si ça ne passe pas et la tolérance permise par la troncation puis tu sommes les deux pour donner la tolérance vraiment autorisée. Si tu as une meilleure idée pour prévenir et compute les n_steps nécessaires, précise le moi puis on en discute mais implémentes ça d'abord

- [x] Reimplement only `unroll_certificate` on top of PR 15.
- [x] Keep the Optyx regression tests minimal.
- [x] Use Optyx from the Marimo experiments in `content/photonic`.
- [x] Restructure the short algorithm and its rigorous appendix.
- [x] Run the required Optyx and wiki checks.
