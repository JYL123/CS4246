import os
import pandas as pd
import pickle

file_size = os.stat("/Users/ljy/CS4246/data/value.txt").st_size
if file_size != 0:
    my_file_handle=open("/Users/ljy/CS4246/data/value.txt","r")
    txt = my_file_handle.read()
    print(txt)
else:
    print("empty data")

new_file=open("/Users/ljy/CS4246/data/value.txt",mode="w",encoding="utf-8")
fruits=["3.6\n","7.4\n","7.8\n"]
new_file.writelines(fruits)
new_file.close()

file_size = os.stat("/Users/ljy/CS4246/data/value.txt").st_size
if file_size != 0:
    with open("/Users/ljy/CS4246/data/value.txt") as file:
        for line in file:
            # preprocessing line
            line = line.strip()
            for number in line.split():
                print(float(number))

new_file=open("/Users/ljy/CS4246/data/value.txt",mode="w",encoding="utf-8")
fruits=["3.6\n","7.4\n","7.8\n"]
new_file.writelines(fruits)
new_file.close()

# one_line_dict = exDict = {1:1, 2:2, 3:3}
# df = pd.DataFrame.from_dict([one_line_dict])
# df.to_csv('file.txt', header=False, index=True, mode='a')

a = {
  'a': 1,
  'b': 2
}

with open('file.txt', 'wb') as handle:
  pickle.dump(a, handle)

with open('file.txt', 'rb') as handle:
  b = pickle.loads(handle.read())

print(a == b) # True