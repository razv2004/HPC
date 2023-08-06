import csv
import random

from hailo_record import HailoRecord

file = open('HailoData.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)

records = []
i = 0
for row in csvreader:
    print(i)
    i += 1
    record = HailoRecord(row)
    records.append(record)

print(len(records))
for _ in range(10):
    print(records[random.randint(0, 1000)])
