数据集名称 
HPE-360
The fisheye-distorted version of three popular public benchmark datasets.

数据集描述
我们在三个开源且流行的头部姿态估计数据集，BIWI，300W-LP，AFLW2000，上创建了对应的鱼眼失真版本。我们将新的鱼眼失真数据集命名为BIWI-360，300W-LP-360，AFLW2000-360，并将三个鱼眼失真数据集打包命名为HPE-360。我们使用相同的映射函数去执行原始图片到鱼眼失真图像的转换，同时对于鱼眼失真版本的头部姿态仍使用原始的头部姿态groundtruth。

数据来源
（说明数据的来源，包括数据收集的时间、地点和方法。）
由于鱼眼畸变数据集是已有的公开数据集的合成版本，因此只对鱼眼畸变的生成方式进行说明。对于原数据集中的直线图像，首先将人脸区域裁剪出来，然后创建一个五倍于人脸区域高和宽的画布，将人脸区域的中心点随机放置在画布的某个位置P处，P由极坐标rho和theta来表示。rho和theta分别从（0，0.8）以及（-180，180）的均匀分布中采样。接着使用特定的映射函数生成鱼眼畸变效果。最后将畸变后的人脸区域从画布上裁剪出来。

数据集结构
HPE-360
	AFLW2000-360
		image00002.jpg
		....
		label_aflw2000.txt
	BIWI-360
		01
		....
		label_biwi.txt
	300W-LP-360
		AFW
		AFW_Flip
		HELEN
		...
		label_300wlp.txt
	README

数据字段说明
（如果数据集包含表格数据，列出每个字段的名称和含义。）
AFLW2000-360 的标注文件为 label_aflw2000.txt 
BIWI-360 的标注文件为 label_biwi.txt
300W-LP-360 的标注文件为 label_300wlp.txt
txt文件中每行的内容为：图片路径   yaw   roll   pitch   rho   theta(°)
（注：rho为归一化径向位置，其范围为0-0.8）

数据集规模
AFLW2000-360：2000张图片
BIWI-360：6个女性，14个男性，共 15678张图片
300W-LP-360：共 122450 张图片

数据集使用许可和版权信息
（说明数据集的使用许可类型，以及任何相关的版权信息。）
如果使用该数据集请引用
@article{li2024location,
  title={Location-guided Head Pose Estimation for Fisheye Image},
  author={Li, Bing and Zhang, Dong and Huang, Cheng and Xian, Yun and Li, Ming and Lee, Dah-Jye},
  journal={arXiv preprint arXiv:2402.18320},
  year={2024}
}

数据集版本
（v1.0）

数据集更新历史
（如果适用，提供数据集的更新历史记录。）

如何使用数据集
（提供使用数据集的基本指南，包括任何必要的预处理步骤。）
鱼眼畸变版本的头部姿态估计数据集HPE-360中的图片仅包含畸变后的人脸区域，因此无需再进行人脸检测，但是由于数据集中图片的大小不一致，因此需要一些必要的预处理操作。

依赖和环境要求
（如果数据集的使用依赖于特定的软件或环境，提供这些信息。）

常见问题解答（FAQ）
（列出一些用户可能遇到的常见问题及其解决方案。）

贡献者和致谢
（列出对数据集有贡献的个人或组织。）

联系方式
（提供数据集维护者的联系方式，以便用户在遇到问题时能够寻求帮助。）

示例代码
（如果适用，提供一些示例代码，帮助用户快速开始使用数据集。）
