# Visible-light-Infrared-SAR-radar-image-processor
一个用于处理可见光、红外、SAR雷达图像的工具
可见光处理方法：
* 直方图均衡化
* CLAHE增强
* 伽马校正
红外图像处理方法：
* 高斯滤波
* 中值滤波
SAR雷达图像处理方法：
* 小波去噪（使用skimage库）
* 非局部均值去噪
