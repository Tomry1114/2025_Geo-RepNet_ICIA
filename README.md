# Geo-RepNet: Geometry-Aware Representation Learning for Surgical Phase Recognition in Endoscopic Submucosal Dissection

本项目基于 PaddleClas 框架，扩展加入了深度信息，用于处理ESD 图像多分类任务，提升模型在具有深度信息的场景下的表现。

## 论文链接
ICIA
[Google Scholar 链接](https://arxiv.org/abs/2507.09294)

## 代码修改
1.加入深度信息，dataloader支持深度信息的导入，灰度图扩张一个channel 变成三通道与rgb相同的形式导入，采用同样的数据处理方式比如进行标准化等，测试的py文件如下：
https://github.com/Tomry1114/2025_Geo-RepNet_ICIA/blob/main/test_dataloader.py

2.对评价指标进行添加：F1和AUC
2025_Geo-RepNet_ICIA/ppcls/metric/metrics.py

3.tr文件里面有多分类数据集的txt文件

4.里面增加EMA 和 geo的模块代码位置
2025_Geo-RepNet_ICIA/ppcls/arch/backbone/base/theseus_layer.py

5.EMA和geo的测试model.pt 放在checkpoint中
