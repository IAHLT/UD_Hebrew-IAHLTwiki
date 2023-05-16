import trankit, sys, pathlib

d = pathlib.Path(sys.argv[1])

if d.exists():
	print("Error: directory %r exists! Exiting..." % d, file=sys.stderr)
	sys.exit(1)

fpath = "../%s-%%s.%%s" % d.stem

missing = [p for p in [pathlib.Path(fpath % (x, y)) for x in ["train", "dev"] for y in ["conllu", "txt"]] if not p.exists()]
if len(missing) > 0:
	print("Error: missing files %s. Exiting..." % ", ".join(missing), file=sys.stderr)
	sys.exit(1)
	

tasks = ['tokenize', 'mwt', 'lemmatize', 'mwt', 'posdep']

configs = [
    {
    'category': 'customized-mwt', # pipeline category
    'task': task, # task name
    'save_dir': d, # directory for saving trained model
    **{'{x}_{y}_fpath'.format(x, y): fpath % (x, y) for x in ["train", "dev"] for y in ["conllu", "txt"]}
    } for task in tasks
]

for config in configs:
	trankit.TPipeline(training_config=config).train()

trankit.verify_customized_pipeline(
	category='customized-mwt',
	save_dir=d,
	embedding_name='xlm-roberta-base'
)

p = trankit.Pipeline(lang='customized-mwt', cache_dir=d)

