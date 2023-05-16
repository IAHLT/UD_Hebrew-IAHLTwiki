import trankit, sys, fileinput

if sys.argv[1] == "--presegmented":
	presegmented = True
	sys.argv.pop(1)
else:
	presegmented = False


d = sys.argv.pop(1)

#trankit.verify_customized_pipeline(
#	category='customized-mwt',
#	save_dir=d,
#	embedding_name='xlm-roberta-base'
#)

with open("/dev/null", "w") as sys.stdout:
	p = trankit.Pipeline(lang='customized-mwt', cache_dir=d)

sys.stdout = open("/dev/stdout", "w")

if presegmented:
	for line in fileinput.input():
		print("# text = %s" % line, end="")
		print(trankit.trankit2conllu(p(line, is_sent=True)), end="")
else:
	x = p("".join(l for l in fileinput.input()))

	for s in x['sentences']:
		print("# text = %s" % s['text'])
		print(trankit.trankit2conllu(s), end="")

