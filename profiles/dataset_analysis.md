Your hypothesis is confirmed more strongly than most would expect:

97.2% of all ratings are inside batch-rating sessions. The median user has literally 100% of their ratings inside 30-minute clusters (p50 = p90 = p99 = 1.0). Only 2.8% of ratings appear as isolated single events between sessions.

A few things worth sitting with:

The extreme tail. One user submitted 2697 ratings in a single 30-minute window. That's not a watch-then-rate pattern — that's someone logging their entire lifetime of films at account creation. Sessions of 100+ are common. The session-size histogram follows a heavy power-law tail.

The "good correlation" concern you raised is smaller than it looks. Since 97.2% of ratings are in sessions, a watch → review → recommendation → watch → review chain that happens to complete within 30 minutes would get shuffled. But that requires two complete watch cycles within 30 min — unlikely for feature films. You could tighten to 5-10 minutes to preserve those while still capturing the bulk of batch-logging, or only shuffle sessions above some size threshold (e.g. size ≥ 5, which still covers the majority of the data).

The implication for the quality-by-seqlen analysis. The monotonic NDCG drop with sequence length we saw earlier might be partially explained by this: long-sequence users are exactly the ones who batch-logged everything at once. Their "sequence" has no temporal meaning at all — it's just a bag of movies in arbitrary order that the model is trying to find causal structure in. Training on shuffled versions of those sessions might actually narrow the quality gap we saw.

Whether to implement the shuffle augmentation is now clearly worth trying.