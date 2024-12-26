# recsys_eq
Producers Equilibria and Dynamics in Engagement-Driven Recommender Systems

Abstract


Online platforms such as YouTube or Instagram heavily rely on recommender systems to decide what content to show to which users. Producers often aim to produce content that is likely to be shown to users and have the users engage. To do so, producers try to align their content with the preferences of their targeted user base. In this work, we explore the equilibrium behavior of producers who are interested in maximizing user *engagement*. We study two variants of the content-serving rule for the platform's recommender system, and provide a structural characterization of producer behavior at equilibrium: namely, each producer chooses to focus on a single embedded feature.
We further show that specialization, defined as different producers optimizing for different types of content, naturally arises from the competition among producers trying to maximize user engagement. We provide a heuristic for computing equilibria of our engagement game, and evaluate it experimentally. We highlight how i) the performance and convergence of our heuristic, ii) the level of producer specialization, and iii) the producer and user utilities at equilibrium are affected by the choice of content-serving rule and provide guidance on how to set the content-serving rule to use in engagement games.
Organization:

- The core classes and helper functions are in the `source` directory.
- The `ProducersEngagementGame` in `Producers.py` instanciates an Engagement Game and requires the number of producers, the content serving rule (softmax or linear),  and a `user` object(see `Users.py`).
- The workflow involves creating an ProducersEngagementGame object, then calling `best_response_dynamics` on it, all the scripts in the `scripts` directory follow this workflow.
- We use the following values for the random seeds for user embedding generation [13, 17, 19, 23, 29], dimension of embeddings = [5, 10, 15, 20] and the number of producers [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] (see `configs/config_seedproddim.yml`)
- We first cache user embeddings for the various datasets using `src/embs_ml100k.sh`,  `src/embs_amznmusic.sh`, `src/embs_rentrunway.sh`, `src/embs_synthdata.sh`
- We run experiment 1 for number of iterations using the scripts `jobarr_*.sh` in `numiter-runs-exp1`, the results are saved in `numiters_savedframe`
- For experiment 2 for producer distribution and utility we run the scripts `jobarray_*.sh` in `pdud-utils-exp2`, the results are saved in `saved_frames`
- To generate the plots we use the jupyter notebooks in `notebooks`, and all the figures in the paper are in the `plots` folder.

The Conda environment is available in `recsys_eq.yml`