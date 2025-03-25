def mt_run(thread):
	[t.start() for t in thread]
	[t.join() for t in thread]