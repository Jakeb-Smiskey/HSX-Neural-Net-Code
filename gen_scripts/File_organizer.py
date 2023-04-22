import os

basePath = "/media/smiskey/USB/main_coil_0/set_Predicted_data_files" #Edit this
newPath = "/media/smiskey/USB/main_coil_0/set_Predicted/" #Edit this
x = os.listdir(basePath)

for i in x:
	inputfile = os.path.join(basePath, i, "input.HSX_aux")
	outputfile = os.path.join(basePath, i, "metric_output.txt")
	make = "mkdir " + newPath + i
	command_1 = "cp " + inputfile + " " + newPath + i
	command_2 = "cp " + outputfile + " " + newPath + i
	os.system(make)
	os.system(command_1)
	os.system(command_2)

