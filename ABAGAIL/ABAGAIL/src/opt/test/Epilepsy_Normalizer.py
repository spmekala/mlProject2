import sklearn.preprocessing as sk
import pandas as pd
import numpy


filename = "temp_coolingrate.csv"
label = "Gender"
instances = ["CR=0.2 training","CR=0.35 training","CR = 0.5 training","CR=0.65 training","CR=0.8 training","CR=0.95 training","CR=0.2 testing","CR=0.35 test","CR = 0.5 test","CR=0.65 test","CR=0.8 test","CR=0.95 test"]
row_vector = list(instances)
print(row_vector)
print(instances)

file = pd.read_csv(filename, usecols=row_vector)
temp3 = sk.normalize(file)
#for field in instances:
#     temp = file[field]
#     temp1 = sk.normalize(file[field])
#     file[field] = sk.normalize(file)

numpy.savetxt("temp_coolingrate_Normalized.csv", temp3, delimiter=",")

# import sklearn.preprocessing as sk
# import pandas as pd
#
# filename = "epilepsy_NNRO.csv"
# label = "y"
# instances = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39", "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49", "X50", "X51", "X52", "X53", "X54", "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64", "X65", "X66", "X67", "X68", "X69", "X70","X71", "X72", "X73", "X74", "X75", "X76", "X77", "X78", "X79", "X80", "X81", "X82", "X83", "X84", "X85", "X86", "X87", "X88", "X89", "X90", "X91", "X92", "X93", "X94", "X95", "X96", "X97", "X98", "X99", "X100", "X101", "X102", "X103", "X104", "X105", "X106", "X107", "X108", "X109", "X110", "X111", "X112", "X113", "X114", "X115", "X116", "X117", "X118", "X119", "X120", "X121", "X122", "X123", "X124", "X125", "X126", "X127", "X128", "X129", "X130", "X131", "X132", "X133", "X134", "X135", "X136", "X137", "X138", "X139", "X140", "X141", "X142", "X143", "X144", "X145", "X146", "X147", "X148", "X149", "X150", "X151", "X152", "X153", "X154", "X155", "X156", "X157", "X158", "X159", "X160", "X161", "X162", "X163", "X164", "X165", "X166", "X167", "X168", "X169", "X170", "X171", "X172", "X173", "X174", "X175", "X176", "X177", "X178",]
#
# row_vector = instances[:].append(label)
# print(row_vector)
# print(instances)
#
# file = pd.read_csv(filename, usecols=row_vector)
#
# for field in instances:
#     file[field] = sk.normalize(file[field])[0]
#
# file.to_csv("epilepsy_NNRO_Normalized.csv", index=False, header=False)
