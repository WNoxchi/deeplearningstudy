# 2017-Jul-25 15:36 from: https://www.youtube.com/watch?v=2Tw39kZIbhs
import pickle

# pickle out / create pickle
# example_dict = {1:"6", 2:"2", 3:"f"}
#
# pickle_out = open("dict.pickle", "wb")    # "wb": write bytes
# pickle.dump(example_dict, pickle_out)
# pickle_out.close()

# pickle in
pickle_in = open("dict.pickle", "rb")
example_dict = pickle.load(pickle_in)

print(example_dict)
print(example_dict[2])

# NOTE: pickle is serialization in Python. It's useful for anything
# that involves a lot of processing & ends in an object.
# Such as reading in a dataset: csv, files, sql-db, etc.

# NOTE: pickle has no security if being used to transfer data btwn
# servers.
