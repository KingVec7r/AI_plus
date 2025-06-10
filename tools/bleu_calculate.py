from sacrebleu.metrics import BLEU, CHRF, TER

refs = [ # First set of references
         ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
         # Second set of references
         ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']
       ]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = BLEU()
bleu.corpus_score(sys, refs)
 
bleu.get_signature()
 
chrf = CHRF()
chrf.corpus_score(sys, refs)

pass
 