V1版本是最原始的实现  
最开始使用nibabel记载数据，后面又使用simpleitk加载保存数据，导致训练和测试数据有90度偏转。  
stage-1/stage-2的全部实验均是在V1下完成的。效果还可以。

V2版本主要改变了训练集和验证集的加载方式
依然存在V1的问题(nibabel和simpleitk混合使用)。  
数据集加载以病人为单位，不是以slice为单位。验证集的评估方式是3D Dice Score。  
stage-3的全部实验均在V2下完成，效果不好。  
推测是1.病人之间LGE图像差异很大，有可能过拟合了。2.生成fake_lge效果不好

V3版本综合上面两点
统一使用simpleitk，并且在进行cyclegan之前将数据进行预处理。  
包括：1.直方图匹配，将LGE图像的强度分布统一。2.统一resize到512X512再crop到320X320。  
效果还是没有V1好。

拉拉拉  
测试来测试去还是V1好啊，Adam还是厉害，接下来模型融合，weighted diceloss