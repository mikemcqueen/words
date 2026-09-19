• dfs.best.pairs is a receipt for the pair-bonus set used by the last successfully completed dfs.best search.

  It is not the manually maintained best.pairs, and it is not used as an input to the next search. Its main consumer is wf best status.

  ## What it contains

  For a target such as:

  s2/u-cdef/m4/g4

  the workflow computes:

  candidate pairs =
      global classified YES
      ∪ optional target best.pairs

  allowed pairs =
      candidate pairs
      − global classified NO
      − optional target no.pairs

  usable pairs =
      allowed pairs that can be spelled from this target's search bag

  <target>/dfs.best.pairs contains the final usable pairs set: sorted, unique, and target-specific.

  The construction is implemented by build_search_pairs() in workflow/best/state.py:341.

  There are two relevant counts:

  - allowed: pairs remaining after the two NO sets are subtracted, before considering the target’s letters.
  - usable: allowed pairs whose combined letters fit the target’s search bag.

  Only the usable lines are stored in dfs.best.pairs. That is why generation can report something such as:

  Searched with 2 of 300 allowed pairs → s2/u-cdef/m4/g4/dfs.best.pairs

  Here, 300 pairs survived the exclusions, but only two could possibly occur in this target’s DFS search.

  ## How it is created

  During wf best gen … dfs.best, the effective pair set is first built in a temporary directory:

  /tmp/wf-dfs-pairs-XXXX/dfs.best.pairs

  That temporary file is passed to Nutrimatic:

  dfs-anagrams ... --pairs /tmp/wf-dfs-pairs-XXXX/dfs.best.pairs ...

  If the temporary file is empty, the workflow refuses to run dfs.best. Without at least one applicable bonus pair, dfs.best would perform essentially the
  same enumeration as dfs.seed, but without obtaining the intended pair-bonus refinement.

  The ordering after DFS succeeds is important:

  1. dfs-anagrams finishes successfully
  2. its output file is atomically published
  3. the target's dfs.best symlink is published
  4. the effective pair set is published as <target>/dfs.best.pairs

  That code is in workflow/best/generate.py:163.

  dfs.best.pairs is deliberately written last. Suppose DFS is interrupted:

  - Before DFS finishes: neither the new search result nor its pair receipt is published.
  - After the result is published but before dfs.best.pairs is published: status sees the old or missing receipt and considers the search stale.
  - After dfs.best.pairs is published: status knows the recorded pair set belongs to a successfully completed search.

  This prevents an old dfs.best result from being declared current merely because a new pair receipt was written before the new search completed.

  ## How status uses it

  When wf best status needs to evaluate the dfs.best lane, it recomputes today’s expected usable pair set using exactly the same process:

  current YES ∪ current target best.pairs
      − current global NO
      − current target no.pairs
      → filter to the target bag

  It builds that expected set under a temporary directory such as:

  /tmp/wf-usable-pairs-XXXX/dfs.best.pairs

  Then it performs a byte-for-byte comparison against:

  <target>/dfs.best.pairs

  The result is represented internally as:

  (allowed_count, usable_count, current)

  where current means:

  stored dfs.best.pairs exists as a regular file
  and
  stored dfs.best.pairs exactly equals the currently derived usable set

  See Inputs.usable_pairs in workflow/best/state.py:768.

  Because the property is cached, one status calculation for a target does not repeat this construction every time another status row consults it.

  ## Its first status function: detecting no viable refinement

  The _no_usable_pairs status row looks at usable_count.

  If it is zero, status reports:

  no allowed bonus pair fits this target's letters

  and gives detail such as:

  (300 pairs remain after exclusions, none spellable here)

  It does not offer a dfs.best refinement, because generation would refuse that search. Instead it can suggest widening top.segments, reseeding, retracting
  exclusions, or manually adding target best.pairs.

  That logic is in workflow/best/state.py:1017.

  This check is independent of the stored receipt’s contents. It asks whether a worthwhile dfs.best search exists now.

  ## Its second status function: deciding whether dfs.best is stale

  If there is at least one usable pair, best_search_needed evaluates the search:

  1. If dfs.best itself is missing:

     dfs.best missing

  2. If dfs.best exists but the stored and recomputed pair sets differ:

     dfs.best out of date (usable pair set changed)

  3. It also checks other search inputs separately:
      - derived dictionary changed
      - global hard-NO file is newer than dfs.best
      - target-local no.pairs is newer than dfs.best

  The predicate is in workflow/best/state.py:813.

  When this becomes the winning status row, the workflow offers the refine action that reruns the best search and derives a new frontier.

  ## Why content comparison is useful

  The global confirmed-YES file is shared by every target. Suppose another target adds:

  tiger,lily

  but the current target’s bag cannot spell those letters.

  A simple timestamp rule would say:

  classified YES changed after dfs.best
  therefore rerun dfs.best

  That could schedule hours of unnecessary work for every target.

  dfs.best.pairs avoids that. Status recomputes the target-specific usable set:

  stored:     a,b
  recomputed: a,b

  Because they are identical, the dfs.best search is not invalidated by that unrelated YES addition.

  Conversely, if a spellable pair is added manually:

  stored:     a,b
  recomputed: a,b
              b,c

  status knows the previous search did not apply the b,c bonus and reports:

  dfs.best out of date (usable pair set changed)

  Deleting best.pairs, removing a confirmed YES, or deleting a target-local exclusion can similarly be detected through content even when there is no
  useful “deletion timestamp” to inspect.

  ## Why exclusions also have separate timestamp checks

  dfs.best.pairs records only the effective bonus set. It does not record every pair excluded from DFS.

  An exclusion can change search results even if the excluded pair was never in the bonus set. Therefore:

  usable bonus set unchanged

  does not prove:

  the DFS search inputs were unchanged

  That is why global hard-NO and target-local NO files also date dfs.best independently.

  For example, suppose tiger,lily is not in classified YES, so it never appears in dfs.best.pairs. Adding it to no.pairs still tells DFS to reject that
  phrase entirely. The pair receipt remains identical, but the search itself must be treated as stale.

  ## Where it fits in status precedence

  wf best status evaluates ordered rows and displays the first actionable condition. The relevant order is approximately:

  open review
  missing frontier
  review needed
  no usable pairs
  finished DFS newer than frontier
  frontier behind classifications/dictionary
  dictionary maintenance
  next DFS search
  converged

  Therefore a changed pair set does not necessarily produce an immediately visible:

  dfs.best out of date

  For example, if confirmed YES changed after top.segments, the earlier frontier-staleness row may first say:

  top.segments behind its inputs (confirmed-YES set changed)

  Only after the cheaper frontier/review work is settled might the DFS-search row become the next action. This ordering intentionally prefers seconds or
  minutes of frontier work before offering another hours-long DFS run.

  The row ordering is defined in workflow/best/state.py:1146.

  ## What dfs.best.pairs does not do

  It does not:

  - Accumulate pairs for future searches.
  - Feed top-segments; that reads the dfs.best results.
  - Replace best.pairs; best.pairs is an optional manual input.
  - Record the dictionary, exclusions, DFS options, or result contents.
  - Prove that all DFS inputs are current.
  - Restrict DFS results to these pairs; they only receive scoring bonuses.

  Its meaning is narrowly:

  > These are the target-spellable, non-excluded pair-bonus lines passed to the last successfully published dfs.best search.

  One limitation is that this is a textual snapshot. The workflow compares sorted lines, while Nutrimatic normalizes each pair and inserts both word orders
  into its in-memory DfsPairSet. Thus changing a,b to b,a, or storing both orientations, can change dfs.best.pairs even though Nutrimatic may derive the
  same internal scoring keys. That mismatch is particularly relevant if the workflow moves to repeated direct --pairs arguments: the replacement receipt or
  digest should represent Nutrimatic’s normalized effective set, not merely the raw input lines.
