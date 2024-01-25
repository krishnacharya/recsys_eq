# recsys_eq
Producers Equilibria and Dynamics in Engagement-Driven Recommender Systems

Abstract

Online platforms such as YouTube, Instagram, TikTok heavily rely on recommender systems to decide what content to show to which users. Content producers often aim to produce material that is likely to be shown to users and lead them to engage with said producer. To do so, producers try to align their content with the preferences of their targeted user base. In this work, we explore the equilibrium behavior of producers that are interested in maximizing user engagement. We study two variants of the content-serving rule that the platform's recommender system uses, and we show structural results on producers' production at equilibrium. We leverage these structural results to show that, in simple settings, we see specialization naturally arising from the competition among producers trying to maximize user engagement. We provide a heuristic for computing equilibria of our engagement game, and evaluate it experimentally. 
We show i) the performance and convergence of our heuristic, ii) the producer and user utilities at equilibrium,  and iii) the level of producer specialization. 

Organization:

- The core classes and helper functions are in the `source` directory.
- The `ProducersEngagementGame` in `Producers.py` instanciates an Engagement Game and requires the number of producers, the content serving rule (softmax or linear),  and a `user` object(see `Users.py`).
- The workflow involves creating an ProducersEngagementGame object, then calling `best_response_dynamics` on it, all the scripts in the `scripts` directory follow this workflow.
- We use the following values for the random seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29], dimension of embeddings = [5, 10, 15, 20] and the number of producers = np.arange(10, 110, 10).
- The results for number of iterations till Nash equilibrium and utilities at NE are available in the folders `csv_results/br_dynamics` and `csv_results/utility-tables` respectively.
- The pandas Dataframes for each of the 400 instances (of seed x dimensions x producers) for all the experiments are available in the `saved_frames` folder.
- All the figures in the paper are in the `plots` folder.

Conda environment is available in `recsys_eq.yml`