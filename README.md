# KeyPointOfFaceDetection
人脸关键点检测

生成train/test.txt
data_generate1.py、data_generate2.py、data_generate3.py为任务试探，data_generate.py完成最终生成train/test.txt的任务。
data_generate1.py：画出人脸边框以及相应关键点以熟悉程序操作、验证坐标表示以及检验标注数据是否正确。
data_generate2.py：选取原始人脸框的0.25倍进行expand。expand时，注意扩增后的人脸框不要超过图像大小。
data_generate3.py：对人脸进行截取，同时，截取过后，人脸关键点坐标即变为相对于截取后的人脸图了。
data_generate.py：生成train/test.txt的任务。

网络搭建
1. 数据在网络中的维度顺序是什么?<br>
答：N x C x H x W。N个数，C通道数，H高，W宽。
2. nn.Conv2d()中参数含义与顺序?<br>
答：in_channels输入通道数，out_channels输出通道数，kernel_size核大小，stride步长，padding每一条边的补层数， kernel间距，groups从输入通道到输出通道的阻塞连接数，bias是否加入偏值bias，padding_mode表示padding的填充数（默认填0）。
3. nn.Linear()是什么意思?参数含义与顺序?<br>
答：线性变换。in_features每个输入样本的大小，out_features每个输出样本的大小，bias是否加入偏值bias。
4. nn.PReLU()与 nn.ReLU()的区别?示例中定义了很多 nn.PReLU()，能否只定义一个
PReLU?<br>
答：nn.ReLU()是ReLU函数，nn.PReLU()是PReLU函数，PReLU相比ReLU，在x小于等于0时y不再为0，而是一个ax，其中a是一个很小的固定值，如0.01。可以只定义一个PReLU函数，但为了forward计算一步到位，我们一般定义多个激活函数。
5. nn.AvgPool2d()中参数含义?还有什么常用的 pooling 方式?<br>
答：kernel_size核大小，stride步长，padding每一条边的补层数，ceil_mode计算时是否向上取整（默认向下取整），count_include_pad计算时是否包括padding, divisor_override指定除数（否则使用kernel_size作为除数）。
6. view()的作用?<br>
答：把数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor。就是将数据降到一维，做flatten操作。

detector_xzq.py：网络模型与主函数程序
data_xzq.py：数据读取与加载工具程序

完成训练的任务
1. optimizer.zero()与 optimizer.step()的作用是什么？<br>
答：每次计算新的grad时，要把原来的梯度清0。optimizer.zero_grad()可以自动完成这个操作，把所有Variable的grad成员数值变为0，optimizer.step()则在每个Variable的grad都被计算出来后，更新每个Variable的数值。<br>
2. model.eval()产生的效果?<br>
答：model.train()启用BatchNormalization和Dropout，而model.eval()不启用BatchNormalization和Dropout。<br>
3. model.state_dict()的目的是？<br>
答：将每一层与它的对应参数建立映射关系。<br>
4. 何时系统自动进行bp？<br>
答：调用loss.backward()后并且Tensor的requires_grad为True。<br>
5. 如果自己的层需要bp，如何实现？如何调用？<br>
答：通过设置requires_grad参数，训练需要bp的层而冻结其他层。<br>

stage2
stage2-detector-1：做数据增广时，发现翻转与旋转的效果不佳。利用图像饱和度做数据增广。
stage2-detector-2：优化器选用Adam，并在网络中加入BN。
stage2-detector-3：换ResNet18进行训练，最后的fc层需要改成42输出。

stage3
生成非人脸数据，认为与人脸重叠部分的iou<0.3的就是非人脸。
