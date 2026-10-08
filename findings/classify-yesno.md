Q: do i understand the wf complete p2 process correctdly.. it downloads the notes, validates no conflicting checkboxes, extracts the YES and NO words, and directly updates the classified yes/no pairs without checking if a NO pair has previously been classified yes?


A: Mostly right, but the last part is off. complete p2 does check new NO pairs against pairs already classified YES, and it refuses to fold anything if it finds one.

The steps, from workflow/complete.py (P2 = Complete(...)):

1. retrieve (p2_retrieve): downloads the notes as enex files into the bundle.
2. extract (steps/p2_extract.py): parses the notes with --two-checkboxes, so a NO row is one where N was ticked, not one where Y was left blank. It builds the YES and NO sets in /tmp. If any pair is marked both ways, in either word order (a,b vs b,a), it stops with a list of those rows and puts nothing in the bundle. That's the "no conflicting checkboxes" check you described.
3. classify (steps/p2_classify.py:run_step): before either classified set changes, it calls classify.conflict() on both the YES and NO sets.
   - This compares the bundle's pairs with the opposite classified set on disk, in both word orders (classify.contradictions).
   - Any match raises Cannot classify NO: N input pair(s) already classified YES: ... and neither set is folded.
   - Only if both checks pass does it fold YES into classified/yes and NO into classified/no.
4. merge: folds the bundle into the phase's done set.
5. archive and close: move the inputs into p2/done/{in,out} and remove the bundle.

Sentence-scoped bundles (built with eval p2 -s N) follow different rules, from classify.opposing():
- A sentence NO is checked against both the sentence YES and the global YES.
- A sentence YES is checked only against the sentence NO. It's allowed to contradict a global NO, since a pair that's wrong in general can still be right for one sentence.
