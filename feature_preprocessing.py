import glob, os, re, types, argparse
from argparse import RawTextHelpFormatter
import yaml, json
import sys

def expand_features(features):
	'''
	Convert hierarchichal dictionary (nested dictionaries) into a single key->value dictionary
	(This implementation only considers integers, long integers and floats, and lists/dictionaries of these basic types)
	'''
	expanded_features = {}
	for k0,v0 in features.iteritems():
		if type(v0) == types.DictType:
			for k1,v1 in v0.iteritems():
				if type(v1) in [types.IntType, types.FloatType, types.LongType]:
					expanded_features["%s.%s" % (k0, k1)] = v1
				elif type(v1) == types.ListType and len(v1) > 0 and type(v1[0]) in [types.IntType, types.FloatType, types.LongType]:
					for i in range(len(v1)):
						expanded_features["%s.%s_%d" % (k0, k1, i)] = v1[i]
				elif type(v1) == types.DictType:
					for k2,v2 in v1.iteritems():
						if type(v2) in [types.IntType, types.FloatType, types.LongType]:
							expanded_features["%s.%s.%s" % (k0, k1, k2)] = v2
						elif type(v2) == types.ListType and len(v2) > 0 and type(v2[0]) in [types.IntType, types.FloatType, types.LongType]:
							for i in range(len(v2)):
								expanded_features["%s.%s.%s_%d" % (k0, k1, k2, i)] = v2[i]
	return expanded_features

def get_re_pattern(pattern):
	'''
	Convert text pattern to python regular expression syntax
	'''
	pattern = r"^"+pattern+r"$"
	pattern = pattern.replace(".", r"\.")
	pattern = pattern.replace("*", r".*")
	return re.compile(pattern)

def filter_features(features, include=[], remove=[]):
	'''
	Filter features by keeping only features in 'include' and/or removing features in 'remove'
	'''
	filtered_features = features.copy()
	if len(include) > 0:
		filtered_features = []
		for feat in include:
			re_pattern = get_re_pattern(feat)
			filtered_features.extend([(k,v) for k,v in features.iteritems() if re_pattern.search(k) is not None])
		filtered_features = dict(list(set(filtered_features)))
	if len(remove) > 0:
		to_remove = []
		for feat in remove:
			re_pattern = get_re_pattern(feat)
			to_remove.extend([k for k in features.iterkeys() if re_pattern.search(k) is not None])
		for k in to_remove:
			if filtered_features.has_key(k):
				del filtered_features[k]
	return filtered_features

def preprocess_features(collection, include, remove):
	'''
	Load and preprocess features, by:
	(1) converting the input hierarchichal dictionary (nested dictionaries) into a single key->value dictionary,
	(2) filtering the features by keeping only features in "include" and/or removing features in "remove"
	(3) deleting features that do not appear in all tracks (probably due to NANs)
	'''
	features_occurrence = {}
	preprocessed_features = {}
	i = 1
	for filename in sorted(glob.glob("features/%s/*.sig" % collection)):
		trackid = filename[filename.rfind("/")+1:filename.rfind(".")]
		features = yaml.load(open(filename))
		# (1) convert hierarchichal dictionary (nested dictionaries) into a single key->value dictionary
		features = expand_features(features)
		# (2) filter features by keeping only features in 'include' and/or removing features in 'remove'
		features = filter_features(features, include=include, remove=remove)
		preprocessed_features[trackid] = features
		for k in features.iterkeys():
			if not features_occurrence.has_key(k): # count occurrences of the different features
				features_occurrence[k] = 0
			features_occurrence[k] += 1
		if i % 20 == 0:
			print "%d tracks processed" % i
		i += 1
	# (3) Delete features that do not appear in all tracks
	for feat,occ in sorted(features_occurrence.items(), key=lambda x: x[1], reverse=True):
		if occ < len(preprocessed_features):
			for trackid in preprocessed_features.iterkeys():
				if preprocessed_features[trackid].has_key(feat):
					del preprocessed_features[trackid][feat]
	if not os.path.exists("preprocessed_features"):
		os.mkdir("preprocessed_features")
	if not os.path.exists("preprocessed_features/%s" % collection):
		os.mkdir("preprocessed_features/%s" % collection)
	for trackid, feats in preprocessed_features.iteritems():
		json.dump(feats, open("preprocessed_features/%s/%s.json" % (collection, trackid), "w"))

if __name__ == '__main__':
	description = '''
	Load and preprocess features, by:
	(1) converting the input hierarchichal dictionary (nested dictionaries) into a single key->value dictionary,
	(2) filtering the features by keeping only features in "--include-features" and/or removing features in "--remove-features"
	(3) deleting features that do not appear in all tracks (probably due to NANs)
	'''
	parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
	parser.add_argument('collection', help='Collection name (e.g.: majorminer)')
	parser.add_argument('-i', '--include-features', nargs='+', help='Features to include (e.g.: lowlevel.* rhythm.* tonal.* (separated by a whitespace))')
	parser.add_argument('-r', '--remove-features', nargs='+', help='Features to remove (e.g.: lowlevel.mfcc.*')
	args = parser.parse_args()
	preprocess_features(args.collection, args.include_features, args.remove_features)