# RL Search on Graph Environments

1. `bartenv.py, gpt2env.py, t5env.py`: *OpenAI Gym* environments for RL search on sentence similarity graph (H2). The graph uses *numba* for parallelizing large matrix calculations. At each step, the generator generates sentences and all the sentences are added to the graph, updating it. At all steps, `top_k` sentences are only considered by the RL algorithm to take the next step. The parameter `sim_threshold` is the lower bound for the existence of an edge between two sentences.
