import os
import B_field as B


def runVMEC(num):
    # Edit for Aux Coil Input Path
    dirConfigs = "/home/smiskey/Documents/HSX/HSX_Configs/jobs{}/".format(num)
    basePath = "/home/smiskey/Documents/HSX/VMEC"  # Edit for VMEC PATH
    files = ["jxbout_HSX_aux.nc", "mercier.HSX_aux", "parvmecinfo.txt",
             "threed1.HSX_aux", "timings.txt", "wout_HSX_aux.nc"]

    # Gets the location of all input files
    inputFiles = []
    # for i in range(numDir):
    for i in os.listdir(dirConfigs):
        inputFile = os.path.join(dirConfigs, i, "input.HSX_aux")
        inputFiles.append(inputFile)

    # Run VMEC with input files and create new directory for each run
    os.chdir(basePath)
    os.system("mkdir Jobs{}".format(num))
    for i in os.listdir(dirConfigs):
        inputFile = os.path.join(dirConfigs, i, "input.HSX_aux")
        runVMEC = "mpiexec -n 2 ./xvmec " + inputFile
        print(runVMEC)
        command = "mkdir " + "Jobs{}".format(num) + "/" + i
        os.system(command)
        os.system(runVMEC)
        for j in files:
            command = "mv " + j + " Jobs{}".format(num) + "/" + i  # Likewise ^
            os.system(command)


def mainCoil_edits(num):
    dirConfigs = "/home/smiskey/Documents/HSX/HSX_Configs/jobs{}/".format(num)

    # Replaces the main coil current with a new current that makes the on axis B field 1T in the input file
    for i in os.listdir(dirConfigs):
        inputFile = os.path.join(dirConfigs, i, "input.HSX_aux")
        f = open(inputFile, 'r')
        lines = f.readlines()
        f.close()
        newlines = lines
        cur = str(B.B_field(num, i))  # This is i+1 due to my file formatting
        # This only works if main coil current is starting at base of -10722.0	count += 1
        newlines[22] = newlines[22].replace(" -10722.0 ", cur + " ")
        # Does not break after re-running VMEC with new coil current!!
        # Can't replace something that isn't there
        print(newlines[22])
        f = open(inputFile, 'w')
        for l in newlines:
            f.write(l)
        f.close()


coilcurrent = input("Enter Coil Current Number: ")
runVMEC(coilcurrent)
mainCoil_edits(coilcurrent)
