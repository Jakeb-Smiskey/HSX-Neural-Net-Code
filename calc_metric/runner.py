import run_calcsnew as cal
import os
dirRun = "/media/smiskey/USB/main_coil_0/set_Predicted_data_files" #Edit this
print(os.listdir(dirRun))
for i in os.listdir(dirRun):
    print(i)
    cal.run(i, dirRun)
#cal.run("job_364", dirRun)


