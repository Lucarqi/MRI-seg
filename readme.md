# Cyclegan Plus U-net
使用cyclegan数据增强，使用U-net分割

### v1
训练数据和验证数据是从C0LGE+T2LGE+LGE中抽取的，而且验证评价是二维Dice Score。

### v2
训练数据和验证数据以病人为划分，验证时使用三维Dice Score，使用HD，SD等。