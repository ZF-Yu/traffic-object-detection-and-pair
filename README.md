## 2020百度之星开发者大赛：交通标识检测与场景匹配 - 第十八名解决方案
****************************************************************************************************************************************
- **【赛事信息】** [2020百度之星开发者大赛：交通标识检测与场景匹配](https://aistudio.baidu.com/aistudio/competition/detail/39)
- **【参赛队伍】** 3个小白顶个吴恩达    
- **【初赛成绩】** F1 Score：0.54676 【14/140】     
- **【复赛成绩】** F1 Score：0.59972 【18/50】    
- **【团队成员】** 李想[@似水5494264](https://blog.csdn.net/tiancailx)(同方威视技术股份有限公司)、
                  于子锋[@诺艾尔和阿梓喵](https://github.com/nuoaier)、
                  杨航[@绝对灬尖刀ok](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/315398)(上海汽车变速器有限公司)

****************************************************************************************************************************************
### 模型文件
- **【模型说明】** 本次比赛参数文件已上传百度云盘      
- **【提取路径】** 链接：https://pan.baidu.com/s/1TaKGIFFeZzH-59-AVeOE5g  
- **【提取码】** qwkh      

****************************************************************************************************************************************
### 系统环境
- **python 3.6.5**     
- **paddlepaddle-gpu 1.8.0**   
- **opencv-python 3.3.0.10**      
- **opencv-contrib-python 3.3.0.10**   
- **numpy 1.17.5**   
****************************************************************************************************************************************
### 方案介绍

  - 本次比赛涉及的技术方面很广泛，在Baseline的框架基础上，将比赛分为两个阶段，本项目在基线的基础上进行修改，利用目标检测开发套件PaddleDetection与度量学习库（metric_learning）进行开发。   
    
  - 检测阶段使用faster_rcnn_ResNeXt101_vd_FPN模型做迁移学习，采用kmeans法首求出锚框大小的大小分布，在配置文件中将anchor_sizes改为[6,12,18,28,48]并将anchor_start_size改为6 
    
  - 修改nms的score_threshold为0.07；求出图像的均值和方差分别为[0.4255, 0.4542, 0.4393]和[0.261276, 0.2659, 0.27744] 
    
  - 细分类和匹配部分，将image_shape改为[3,128,128]。匹配部分采用resnet152网络，通过triplet loss进行匹配训练, 求得输入图像大小的均值和方差，分别为[0.473,0.4997,0.49139]和[0.26557,0.2672,0.278],在imgtool文件修改
    
  - 去掉随机裁剪，并将interpolation修改为更为广泛使用的线性插值cv2.INTER_LANCZOS4，image_size同为[3,128,128],调试并采用margin为0.7，最后提交成绩
  
  - 主代码请参考main_code.ipynb
    
****************************************************************************************************************************************
### 详细处理流程 
  - **数据准备**        
  
    【1】 将traffic数据集存放在某一个路径下，然后建立train和test两个目录，分别存放训练集和测试集，建立data/traffic_data/tag文件夹    

    【2】 解压train1和train2到data/traffic_data文件夹下，将train1放入data/traffic_data文件夹下
    
    【3】 按照相同操作将train_label解压并将train_label文件夹放入data/traffic_data/tag/train文件夹下

    【4】 将test压缩包解压至data/traffic_data/文件夹下

    【5】 将train2文件夹中的文件逐一移动到train文件夹中
    
**************************************************************************************************************************************** 
  - **检测训练与验证**        
  
    【1】 我们使用了COCO数据集上训练好的faster_rcnn_ResNeXt101_vd_FPN_1x模型做迁移学习，调参训练后mAP值可以超过0.8    

    【2】 因为小目标较多，因此我们采用FPN并采用kmeans法首先求出锚框大小的大小分布，将配置文件work/PaddleDetection_traffic/configs/traffic/faster_rcnn_x101_vd_64x4d_fpn_1x.yml和work/PaddleDetection_traffic/configs/traffic/faster_rcnn_x101_vd_64x4d_fpn_1x_test.yml中的anchor_sizes改为[6,12,18,28,48]并将anchor_start_size改为6
    
    【3】 调试并修改nms的score_threshold为0.07；求出图像的均值和方差，并在work/PaddleDetection_traffic/configs/traffic/faster_fpn_reader.yml文件中修改

    【4】 将gamma改为[0.3,0.1,0.01]，学习率以及学习率衰减同基线，最后一次衰减减少1万次迭代，共迭代11万次，milestones为[60000，80000，100000]。分别在faster_rcnn_x101_vd_64x4d_fpn_1x.yml和 faster_rcnn_x101_vd_64x4d_fpn_1x_test.yml中的配置文件修改

    【5】 验证结果：复赛训练集很大，不建议在复赛训练阶段加入验证，时间较长。验证阶段，在配置文件中将save_prediction_only=true时，将会直接生成检测结果的文件，存放在detect文件夹中
    
**************************************************************************************************************************************** 
  - **细分类训练及匹配**        
  
    【1】 下载ResNet152度量学习预训练模型    

    【2】 对数据集进行细分类训练，能够进行匹配的样本作为一类，在work/metric_learning_traffic/train_elem.py中将image_shape改为[3,128,128]训练18000迭代，细分类训练部分对最终结果影响较小
    
    【3】 基于训练好的模型进行finetune，通过triplet loss进行匹配训练, 求得数据集图像大小的均值和方差。去掉了随机裁剪，并将interpolation修改为更为广泛使用的线性插值cv2.INTER_LANCZOS4在work/metric_learning_traffic/imgtool.py中修改

    【4】 在work/metric_learning_traffic/train_pair.py文件中修改image_size为[3,128,128],调试并采用margin为0.7

    【5】 该阶段将检测结果进行匹配并保存结果在/home/aistudio/work/metric_learning_traffic/output/result中，采用resnet152网络，image_size为[3,128,128],其他参数同基线。
    
**************************************************************************************************************************************** 
### 成绩记录        

 将生成的detect文件上传，最终得到初赛test数据集0.54676和复赛test数据集0.59972的F1 Score

### 参考文献
[1] PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection) 

[2] metric learning (https://aistudio.baidu.com/aistudio/projectdetail/169466)

[3] resnext (https://arxiv.org/abs/1611.05431)

[4] Faster RCNN(https://arxiv.org/abs/1506.01497)

### 联系方式

李想[@似水5494264](https://blog.csdn.net/tiancailx)(同方威视技术股份有限公司)、
于子锋[@诺艾尔和阿梓喵](https://github.com/nuoaier)、
杨航[@绝对灬尖刀ok](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/315398)(上海汽车变速器有限公司)         
    
