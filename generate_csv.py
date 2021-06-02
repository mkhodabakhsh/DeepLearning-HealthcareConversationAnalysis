
from otter2csv import *
cwd = os.getcwd()
print("CHECKPOINT" , cwd)
out = generate_otter_csv(cwd)
print(out)