# Cyclegan Plus U-net
使用cyclegan数据增强，使用U-net分割

### v1
训练数据和验证数据是从C0LGE+T2LGE+LGE中抽取的，而且验证评价是二维Dice Score。

### v2
训练数据和验证数据以病人为划分，验证时使用三维Dice Score，使用HD，SD等。  
效果没有v1好，我猜测有以下的原因：  
1.生成的fake LGE效果并不是很好  
2.验证集使用三维Dice Score，得到的结果与数据集有很大关系(不同病人图像之间有很大差别)，过拟合了。