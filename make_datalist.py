import glob
import os.path as osp
import os
import csv

def main():
#     path = "../ManagerOffice_bl1_fordemo"
#     datalist = glob.glob(path +"/GT/*")
#     datalist.sort()
#
#     with open("MI_TRAIN.csv", 'w', newline='') as csv_file:
#         for idx,data_path in enumerate(datalist):
#             if idx % 10 == 0:
#                 continue
#             dir_name = osp.basename(data_path)
#             data_list = [dir_name]
#             for v in range(0,7):
#                 for u in range(0,7):
#                     data_list.append("{}_{}.png".format(str(v), str(u)))
#             w = csv.writer(csv_file)
#             w.writerow(data_list)
#     with open("MI_VAL.csv", "w", newline='') as csv_file:
#         for idx,data_path in enumerate(datalist):
#             if idx % 10 != 0:
#                 continue
#             dir_name = osp.basename(data_path)
#             data_list = [dir_name]
#             for v in range(0,7):
#                 for u in range(0,7):
#                     data_list.append("{}_{}.png".format(str(v), str(u)))
#             w = csv.writer(csv_file)
#             w.writerow(data_list)
        basepath = "../"
        paths = ["LF_DATA"]
        with open("LF_DATA_TRAIN.csv", 'w', newline='') as csv_file:
            for path in paths:
                full_path = basepath + path
                datalist = glob.glob(full_path + "/BACK/*")
                datalist.sort()
                for idx, data_path in enumerate(datalist):
                    dir_name = osp.basename(data_path)
                    data_list = [full_path+"/GT/"+dir_name[:-1]+"0", data_path]
                    for v in range(0,7):
                        for u in range(0,7):
                            data_list.append("{}_{}.png".format(str(v+2), str(u+2)))
                    w = csv.writer(csv_file)
                    w.writerow(data_list)

        with open("LF_DATA_VAL.csv", 'w', newline='') as csv_file:
            for path in paths:
                full_path = basepath + path
                datalist = glob.glob(full_path + "/BACK/*")
                datalist.sort()

                for idx, data_path in enumerate(datalist):
                    if idx % 10 != 0:
                        continue
                    dir_name = osp.basename(data_path)
                    data_list = [full_path+"/GT/"+dir_name[:-1]+"0", data_path]
                    for v in range(0,7):
                        for u in range(0,7):
                            data_list.append("{}_{}.png".format(str(v+2), str(u+2)))
                    w = csv.writer(csv_file)
                    w.writerow(data_list)





main()
