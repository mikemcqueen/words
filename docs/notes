18-Apr-2026

Based on below-the-line CoT, I think that ~2mo/S is reasonable but ~6.5mo/S is not, for the first S.

2mo is reasonable because during that ~2mo first automated run of S1 (or whichever i choose first),
I will also be working on:
  a) manual review of partial results,
  b) possible 2nd-pass auto-review of top probabilities from dense model 1st-pass (e.g. via thinking
     reponses from a MoE model on Mac)
  c) discovery/development/refinement of review/select/retire workflow and pipeline

  C) is really where the time needs to be spent; it is the most opaque to me; least intuitive; easy
     to be distracted away from; 

---------the line---------

so what is the next step here?  some options:

  * bug fixing
  * minor feature addition
  * perf improvements
  * additional model testing
  * refining/automating model testing/evaluation pipeline
  * refining/automating (as much as possible) the manual pair review/select/retire pipeline
  * doing actual manual pair review

  A Brutal Truth:

  I think that manual pair review is ultimately the work I'm (mostly) avoiding by doing other work,
  with the exception of the other work related to the manual pair pipeline. And in fact I think it
  is arguable that actual manual pair review should not start until that pipeline is (mostly, or at
  least more than currently) in place.

  Unless the explicit determination is, "I have no idea what this is going to look like, and don't
  want to spend any time thinking or guessing about it", in which case, fire off one round of manual
  review in order to get a feel for the shape of the problem first.

  Key takeaway:

  If I'm not doing manual review, or builing the pipeline for manual review, I'm probably avoiding
  doing it.

  Counterpoint:
  
  It would be really really nice to find a performant model for the mac mini (and even PC2) that
  produced meaningful results. The difference between 1M pairs a day and 2.5M pairs a day is 30M/
  month vs. 75M/mo.

  For reference, S1 has 200M min-4-letter combos. That extra processing is the difference between
  between taking ~2.5 months and ~6.5 months for a first-pass auto-filter on just S1 alone. Combos
  for subsequent S2+ should drop significantly due to having already been filtered. 

  Ok now that I see those numbers, let me consider a 4x improvement that I might get from a 3090
  instead of wasting my time on the Mac.

  4M/day = 120M/mo. 200M in 50 days or ~1.5mo. So I knock a month off S1 time for $1000. That'll
  generate .003 * 200M = 600K results for manual review? I'd need to consistently review 12k/day
  to be ready for the next batch in 50 days. I think 1000 in 15 minutes is a conservative estimate,
  so 4k/hr, 12k in 3 hours. Every day. Hmmm... probably not going to happen.

  In any case, 2.5mo with Mac/PC2 is *way* better than 6.5mo. But I also want to avoid being locked
  into a "mass manual review of everything is the best/only way" mindset/approach. That's why work 
  on the pipeline is so important. It is probable/potential/hopeful that in the process of defining
  and implementing the tools and processes needed to reduce the cognitive & processing load of exec-
  uting the workflow, that I also discover & formalize new approaches and "optimizations" to the
  process itself, that may in turn lead to less of a need for, or at least a more directed approach
  to, both automatic and manual review for subsequent S.
