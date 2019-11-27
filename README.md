# KeyPointOfFaceDetection
人脸关键点检测

任务一：生成train/test.txt
data_generate1.py、data_generate2.py、data_generate3.py为任务试探，data_generate.py完成最终生成train/test.txt的任务。
data_generate1.py：画出人脸边框以及相应关键点以熟悉程序操作、验证坐标表示以及检验标注数据是否正确。
data_generate2.py：选取原始人脸框的0.25倍进行expand。expand时，注意扩增后的人脸框不要超过图像大小。
data_generate3.py：对人脸进行截取，同时，截取过后，人脸关键点坐标即变为相对于截取后的人脸图了。
data_generate.py：生成train/test.txt的任务。

任务二：网络搭建
