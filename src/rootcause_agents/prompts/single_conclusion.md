Now give your structured conclusion.

- `hypotheses`: at least two competing hypotheses ranked best first. Each has root_cause_component (exact service name), root_cause_category, a mechanism (claims of kind inference/assumption), supporting AND contradicting evidence IDs, unexplained symptoms and a confidence in [0,1] (sum ≤ 1).
- Check temporal order (cause must precede effect) and do not confuse coincidence with causation.
- `conclusive`: false if the evidence is insufficient to name a root cause (an inconclusive answer is better than a confident wrong one).
- `report`: summary, facts (with evidence IDs), inferences, assumptions and recommended actions (mitigate_now / fix / prevent / investigate_further with priorities).
- Cite only evidence IDs that appeared in tool results.
