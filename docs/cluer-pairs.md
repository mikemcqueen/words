pairs.s7.w.w is "pairs.big" grepped for 'w.*,.*w'.

python cluer/query_index.py -f ../nutrimatic/results/s7/pairs.s7.w.w -j | sort -u  > ../nutrimatic/results/s7/pairs.cluer.s7.w.w


* Main problem with all below, i want to run *ALL POSSIBLE PAIRS* through cluer, not just
  those present in wiki-index.  so i ported the pairs.cpp tool to nutrimatic for this. and
  i forget what i did after that, but i think i generated all s7 pairs.


 history | grep "(idx.2.s7.m4.cluer|cluer.s7.m4)"

 2071  python cluer/query_index.py -f ../nutrimatic/idx/idx.2.s7.m4 > tmp/idx.2.s7.m4.cluer

 2072  comm -12 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/idx.2.s7.m4.cluer  | wc -l
 2073  mv tmp/idx.2.s7.m4.cluer tmp/cluer.s7.m4

 2078  python cluer/query_index.py -f ../nutrimatic/idx/idx.2.s7.m4 -j > tmp/idx.2.s7.m4.cluer.adj

 2079  mv tmp/idx.2.s7.m4.cluer.adj tmp/cluer.s7.m4.adj

 2081  comm -12 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2082  less tmp/cluer.s7.m4.adj
 2085  pcomm -12 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2086  comm -12 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2087  comm -23 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2088  pcomm -23 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2089  pcomm -13 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2090  comm -13 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | wc -l
 2091  pcomm -13 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | less
 2093  pcomm -13 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.adj  | less

 2116  python cluer/query_index.py -f ../nutrimatic/idx/idx.2.s7.m4 -j -e > tmp/idx.2.s7.m4.cluer.exact
 2117  python cluer/query_index.py -f ../nutrimatic/idx/idx.2.s7.m4 -j -e > tmp/idx.2.s7.m4.cluer.exact -h
 2122  python cluer/query_index.py -f ../nutrimatic/idx/idx.2.s7.m4 -e > tmp/idx.2.s7.m4.cluer.exact

 2125  mv tmp/idx.2.s7.m4.cluer.exact tmp/cluer.s7.m4.exact
 2145  wc -l tmp/cluer.s7.m4*

 2148  pcomm -12 ../nutrimatic/idx/seed.s7.m4.idx2.85.15.pairs tmp/cluer.s7.m4.exact  | wc -l

might be subtle issue with this, see notebook/todo.

 2169  python -m src.filter --yes  --pm .90 --pr .1 -d final/.wf/p1/done/out   ../nutrimatic/idx/idx.2.s7.m4 > tmp/seed.s7.90.10
