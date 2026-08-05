en fait, regarde les Prs (les commits récents dessus des agents et de giovanni), il y a très peu de nouvelles classes et méthodes à coder. Il faut rester dans le fichier déjà existrant, aller dans les méthodes pré ecistantes et y introduire la théorie. En l'occurence, je pense que si tu lis les changements récents qui ont eu lieu, il faut quasiment ne changer que la méthode fix pour qu'à une tolérance donnée, avec ou sans loss; avec truncation_max (chi) et n_steps_max, puisse compute le fix point à partir des n steps données pasr la théorie (le contenu qu'il y a dans le dossier photonique dans le wiki). Sois minimaliste, rigoureux et concis, il faut y introduire la théorie de manière nécessaire et suffisante. Copie tes changements actuels, rebase sur la branche des Prs puis recode ce que tu estimes essentiel aux méthodes incomplètes et SEULEMENT si tu le juges primordial pour la propreté du codes, ajoute quelques fonctions.

- [x] Rebase the work on PR26 and replace the loss-only depth with the stationary boson-sampling certificate.
- [x] Keep the implementation inside existing methods/files and preserve the PR26 API.
- [x] Test certified lossless and lossy depths, `max_steps`, and `chi` through Optyx diagrams.
- [x] Run the fixpoint and feedback suites, the channel doctests, and `pflake8 optyx`.

Mathematical design: for a passive photonic feedback step with loop block
`U_ll`, fresh Fock occupation bounded by `qbar`, and round-trip
transmissivity `gamma = 1 - loss`, choose the smallest `k <= max_steps` such
that `4 K(qbar) sum(arcsin(gamma ** (k / 2) sigma(U_ll ** k)) ** 2) <= tol`.
The calculation uses only the one-step path matrix, its Fock creations, and an
`L x L` loop block; `fix` then contracts `at_time(k - 1)` exactly once.
