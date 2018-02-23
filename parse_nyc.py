import sys
import csv
import collections


user_label = collections.defaultdict(int)
with open('metadata') as md:
	for line in md.readlines():
		vec = line.split()
		user_label[vec[0]] = int(vec[3])

with open('nyc_data.csv', 'wb') as nyc:
	writer = csv.writer(nyc)
	with open('reviewContent') as rc:
		for line in rc.readlines():
			vec = line.split()
			label = user_label[vec[0]]
			sentence = ' '.join(vec[3:])
			writer.writerow([sentence, label])

