title: TLD算法学习
date: 2016/12/30 22:04:12
categories:
- 计算机视觉
tags:
- 目标跟踪
- oepncv
- 代码
- TLD
---
[TOC]

2010年发表的论文《Tracking-Learning-Detection》 , GitHub上有很多C++版本的TLD，比如[arthurv](https://github.com/arthurv/OpenTLD)，注释比较详细，但速度很慢。

```

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-f518840d2338852d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
# TLD算法的构成

TLD算法主要由三个模块构成：追踪器（tracker），检测器（detector）和机器学习（learning）。TLD算法成功的原因就在于它将检测器和跟踪器有机的整合在一起，从而实现了长线跟踪。

TLD是对视频中未知物体的长时间跟踪的算法。**“未知物体”指的是任意的物体**，在开始追踪之前不知道哪个物体是目标。**“长时间跟踪”又意味着需要算法实时计算，在追踪中途物体可能会消失再出现**，而且随着光照、背景的变化和由于偶尔的部分遮挡，物体在像素上体现出来的“外观”可能会发生很大的变化。从这几点要求看来，单独使用追踪器或检测器都无法胜任这样的工作。所以作者提出把追踪器和检测器结合使用，同时加入机器学习来提高结果的准确度。

![](http://johnhany.net/wp-content/uploads/2014/05/tld.png)

<!--more-->

**追踪器**的作用是跟踪连续帧间的运动，当物体始终可见时跟踪器才会有效。追踪器根据物体在**前一帧已知的位置估计在当前帧**的位置，这样就会产生一条物体运动的轨迹，从这条轨迹可以**为学习模块产生正样本**（Tracking->Learning）。


**检测器**的作用是估计追踪器的误差，如果误差很大就改正追踪器的结果。
1. 检测器对每一帧图像都做全面的扫描，找到与目标物体外观相似的所有位置，从检测产生的结果中产生**正样本和负样本**，交给学习模块（Detection->Learning）。
2. 算法从所有**正样本中选出一个最可信的位置**作为这一帧TLD的输出结果，然后用这个结果**更新追踪器的起始位置**（Detection->Tracking）

**学习模块**根据追踪器和检测器产生的正负样本，迭代训练分类器，改善**检测器的精度**（Learning->Detection）。


# 追踪模块

TLD使用作者自己提出的**Median-Flow光流**追踪算法，采用的是**Lucas-Kanade追踪器**

作者假设一个“好”的追踪算法应该具有**正反向连续性**（forward-backward consistency），即无论是按照时间上的正序追踪还是反序追踪，产生的轨迹应该是一样的。作者根据这个性质规定了任意一个追踪器的**FB误差（forward-backward error）**：从时间t的初始位置x(t)开始追踪产生时间t+p的位置x(t+p)，再从位置x(t+p)反向追踪产生时间t的预测位置x`(t)，**初始位置和预测位置之间的欧氏距离**就作为追踪器在t时间的FB误差。

![](http://images.cnitblog.com/blog/460184/201501/271556017069891.png)

## 跟踪点的选择 

前面提到TLD跟踪的不是关键点，它跟踪的是更简单的点：能稳定存在的点，那哪些点是稳定的呢？Median-Flow tracker的基本思想是，看反向跟踪后的残差，用所有点的残差中值作为稳定点的筛选条件。如上图中的黄色点就因为残差太大，被pass掉了，既然稳定点是可以筛选出来的，那么就不必煞费苦心的寻找那些关键点，可以直接将所有的点都作为初始跟踪点，好吧所，有的点毕竟还是太多了，于是作者是选取网格交叉点作为初始跟踪点（见下图框框中黄色的点点）。





在上一帧t的物体包围框里均匀地产生一些点，然后用Lucas-Kanade追踪器**正向**追踪这些**点**到t+1帧，再**反向**追踪到t帧，计算FB误差，筛选出FB误差最小的**一半点**作为最佳追踪点。最后根据这些点的坐标变化和距离的变化计算t+1帧**包围框的位置和大小**（**平移的尺度取中值，缩放的尺度取中值**。取中值的光流法，估计这也是名称Median-Flow的由来吧）



![](http://images.cnitblog.com/blog/460184/201501/271556035971735.png)

可以用NCC（Normalized Cross Correlation，归一化互相关）和SSD（Sum-of-Squared Differences，差值平方和）作为筛选追踪点的衡量标准。(都是越小越好)

NCC: 

![](http://johnhany.net/wp-content/uploads/2014/05/ncc.png)

SSD: 

![](http://johnhany.net/wp-content/uploads/2014/05/ssd.png)


# 学习模块
P-N学习是一种半监督的机器学习算法，它针对检测器对样本分类时产生的两种错误提供了两种“专家”进行纠正：

- P专家（P-expert）：检出漏检（false negative，正样本误分为负样本）的正样本；
- N专家（N-expert）：改正误检（false positive，负样本误分为正样本）的正样本。

## 样本的产生

算法已经确定物体在t+1帧的位置（实际上是确定了相应包围框的位置），从**检测器**产生的包围框中筛选出10个与它距离最近的包围框（两个包围框的交的面积除以并的面积大于0.7），对每个包围框做微小的仿射变换（平移10%、缩放10%、旋转10°以内），产生20个图像元，这样就产生**200个正样本**。再选出若干距离较远的包围框（交的面积除以并的面积小于0.2），产生负样本。这样产生的样本是已标签的样本，把这些样本放入训练集，用于更新分类器的参数。

![](http://johnhany.net/wp-content/uploads/2014/05/structure.png)

作者认为，算法的结果应该具有**“结构性”**：每一帧图像内物体**最多只出现在一个**位置；相邻帧间物体的运动是连续的，连续帧的位置可以构成一条较平滑的轨迹。比如像上图c图那样每帧只有一个正的结果，而且连续帧的结果构成了一条平滑的轨迹，而不是像b图那样有很多结果而且无法形成轨迹。还应该注意在整个追踪过程中，轨迹可能是分段的，因为物体有可能中途消失，之后再度出现。

P专家(修正专家)的作用是寻找数据在**时间上的结构性**，它利用追踪器的结果预测物体在t+1帧的位置。如果这个位置（包围框）被检测器分类为负，P专家就把这个位置改为正。也就是说P专家要保证物体在**连续帧上出现的位置可以构成连续的轨迹**；

N专家(挑一专家)的作用是寻找数据在空间上的结构性，它把**检测器产生的和P专家**产生的所有正样本进行比较，**选择出一个最可信的位置，保证物体最多只出现在一个位置上**，把这个位置作为TLD算法的追踪结果。同时这个位置也用来重新初始化追踪器。

![](http://johnhany.net/wp-content/uploads/2014/05/p-n-experts.png)

比如在这个例子中，目标车辆是下面的深色车，每一帧中黑色框是检测器检测到的正样本，黄色框是追踪器产生的正样本，红星标记的是每一帧最后的追踪结果。在第t帧，检测器没有发现深色车，但P专家根据追踪器的结果认为深色车也是正样本，N专家经过比较，认为深色车的样本更可信，所以把浅色车输出为负样本。第t+1帧的过程与之类似。第t+2帧时，P专家产生了错误的结果，但经过N专家的比较，又把这个结果排除了，算法仍然可以追踪到正确的车辆。


# 检测模块

检测模块使用一个**级联分类器**，对从包围框boundingbox获得的样本进行分类。级联分类器包含三个级别：

 1. **图像元方差**分类器（Patch Variance Classifier）。计算图像元像素灰度值的方差，把方差小于原始图像元方差一半的样本标记为负。论文提到在这一步可以排除掉一半以上的样本。

 2. 集成分类器（Ensemble Classifier）。实际上是一个**随机蕨分类器**（Random Ferns Classifier），类似于随机森林（Random Forest），区别在于随机森林的树中每层节点判断准则不同，而随机蕨的“蕨”中每层只有一种判断准则。所以这个分类器其实不怎么行。
 
![](http://johnhany.net/wp-content/uploads/2014/05/pic6.png)
如上图所示，把左面的树每层节点改成相同的判断条件，就变成了右面的蕨。所以蕨也不再是树状结构，而是线性结构。随机蕨分类器根据样本的特征值判断其分类。从图像元中任意选取两点A和B，比较这两点的像素值，若A的像素大于B，则特征值为1，否则为0。每选取一对新位置，就是一个新的特征值。蕨的每个节点就是对一对像素点进行比较。

>比如取5对点，红色为A，蓝色为B，样本图像经过含有5个节点的蕨，每个节点的结果按顺序排列起来，每个节点表示一个特征。得到长度为5的二进制序列01011，转化成十进制数字11。这个11就是该样本经过这个蕨得到的结果。

同一类的很多个样本经过同一个蕨，得到了该类结果的**分布直方图**。高度代表**类的先验概率p(F|C)**，F代表蕨的结果（如果蕨有s个节点，则共有1+2^s种结果）。

![](http://johnhany.net/wp-content/uploads/2014/05/random-ferns-3.png)

不同类的样本经过同一个蕨，得到不同的先验概率分布。

![](http://johnhany.net/wp-content/uploads/2014/05/random-ferns-4.png)

 以上过程可以视为对分类器的训练。当有新的未标签样本加入时，假设它经过这个蕨的结果为00011（即3），然后从已知的分布中寻找**后验概率最大**的一个。由于样本集固定时，右下角公式的分母是相同的，所以只要找在F=3时高度最大的那一类，就是新样本的分类。只用一个蕨进行分类会有较大的偶然性。另取5个新的特征值就可以构成新的蕨。**用很多个蕨对同一样本分类，投票数最大的类就作为新样本的分类**，这样在很大程度上提高了分类器的准确度。

![](http://johnhany.net/wp-content/uploads/2014/05/random-ferns-5.png)

3. **最近邻**分类器（Nearest Neighbor Classifier）。计算新样本的相对相似度，如大于0.6，则认为是正样本。相似度规定如下：图像元pi和pj的相似度，公式里的N是规范化的相关系数，所以S的取值范围就在[0,1]之间，

![](http://johnhany.net/wp-content/uploads/2014/05/similarity-1.png)

# PN学习半监督学习 
所以，检测器是追踪器的监督者，因为检测器要改正追踪器的错误；而追踪器是训练检测器时的监督者，因为要用追踪器的结果对检测器的分类结果进行监督。用另一段程序对训练过程进行监督，而不是由人来监督，这也是称P-N学习为“半监督”机器学习的原因。

TLD的工作流程如下图所示。首先，检测器由一系列包围框产生样本，经过级联分类器产生正样本，放入样本集；然后使用追踪器估计出物体的新位置，P专家根据这个位置又产生正样本，N专家从这些正样本里选出一个最可信的，同时把其他正样本标记为负；最后用正样本更新检测器的分类器参数，并确定下一帧物体包围框的位置。

![](http://johnhany.net/wp-content/uploads/2014/05/TLD-workflow.png)


# TLD源码理解
TLD算法成功的原因就在于它将检测器和跟踪器有机的整合在一起，从而实现了长线跟踪。

 

## 程序的运行方式 

```cpp
./run_tld -p ../parameters.yml -s ../datasets/06_car/car.mpg -b ../datasets/06_car/init.txt –r
```

1. -p 后面跟的是初始化参数
2. -s 后面的是人工视频的位置
3. -b 是初始化boundingbox的位置  程序用readBB来读取初始化的bounding box

## 程序初始化过程

在run_tld.cpp的main函数里面 进行了配置文件parameters.yml的读取，


### buildGrid(frame1, box);

检测器采用扫描窗口的策略：**扫描窗口步长**为宽高的 10%，**尺度缩放**系数为1.2；此函数构建全部的扫描窗口grid，并计算每一个扫描窗口与输入的目标box的重叠度；重叠度定义为两个box的交集与它们的并集的比；

 

为各种变量或者容器分配内存空间；

 

### getOverlappingBoxes(box, num_closest_init);

此函数根据传入的box（目标边界框），在整帧图像中的全部扫描窗口中（由上面4.1得到）寻找与该box距离最小（即最相似，重叠度最大）的num_closest_init（10）个窗口，然后把这些窗口归入good_boxes容器。同时，把重叠度小于0.2的，归入bad_boxes容器；相当于对全部的扫描窗口进行筛选。并通过BBhull函数得到这些扫描窗口的最大边界。

   

### classifier.prepare(scales);

准备分类器，scales容器里是所有扫描窗口的尺度，由上面的buildGrid()函数初始化；

这是一种典型的特征比较简单，分类器比较复杂的例子.

TLD的分类器有三部分：**方差分类器模块、集合分类器模块和最近邻分类器模块**；这三个分类器是级联的，每一个扫描窗口依次全部通过上面三个分类器，才被认为含有前景目标。这里prepare这个函数主要是初始化集合分类器模块；

集合分类器（随机森林）基于n个基本分类器（共10棵树），每个分类器（树）都是基于一个pixel comparisons（共13个像素比较集）的，也就是说每棵树有13个判断节点（组成一个pixel comparisons），输入的图像片与每一个判断节点（相应像素点）进行比较，产生0或者1，然后将这13个0或者1连成一个13位的二进制码x（有2^13种可能），**每一个x对应一个后验概率P(y|x)= #p/(#p+#n)** （也有2^13种可能），#p和#n分别是正和负图像片的数目。那么整一个集合分类器（共10个基本分类器）就有10个后验概率了，**将10个后验概率进行平均**，如果大于阈值（一开始设经验值0.65，后面再训练优化）的话，就认为该图像片含有前景目标；用的是最简单的blending的组合方法。

后验概率P(y|x)= #p/(#p+#n)的产生方法：初始化时，每个后验概率都得初始化为0；运行时候以下面方式更新：将已知类别标签的样本（训练样本）通过n个分类器进行分类，如果分类结果错误，那么相应的#p和#n就会更新，这样P(y|x)也相应更新了。

pixel comparisons的产生方法：先用一个归一化的patch去**离散化**像素空间，产生所有可能的垂直和水平的pixel comparisons，然后我们把这些pixel comparisons随机分配给n个分类器，每个分类器得到完全不同的pixel comparisons（特征集合），这样，所有分类器的特征组统一起来就可以覆盖整个patch了。

特征是相对于一种尺度的矩形框而言的，**TLD中第s种尺度的第i个特征features[s][i] = Feature(x1, y1, x2, y2); 是两个随机分配的像素点坐标** （就是由这两个像素点比较得到0或者1的） 。计算特征的方法就是求patch在这两个点上的像素的大小。每一种尺度的扫描窗口都含有 totalFeatures = nstructs *  structSize个特征 ；nstructs为树木 （由一个特征组构建，每组特征代表图像块的不同视图表示）的个数；structSize为每棵树的特征个数，也即每棵树的判断节点个数；树上每一个特征都作为一个决策节点；

prepare函数的工作就是先给每一个扫描窗口初始化了对应的pixel comparisons（两个随机分配的像素点坐标）；然后初始化后验概率为0；

 

### generatePositiveData(frame1, num_warps_init);

此函数通过**对第一帧图像的目标框box（用户指定的要跟踪的目标）进行仿射变换来合成训练初始分类器的正样本集**。具体方法如下：先在距离初始的目标框最近的扫描窗口内选择10个bounding box（已经由上面的getOverlappingBoxes函数得到，存于good_boxes里面了，还记得不？），然后在**每个bounding box的内部，进行±1%范围的偏移，±1%范围的尺度变化，±10%范围的平面内旋转，并且在每个像素上增加方差为5的高斯噪声（确切的大小是在指定的范围内随机选择的），那么每个box都进行20次这种几何变换**，那么10个box将产生200个仿射变换的bounding box，作为正样本。具体实现如下：
```
getPattern(frame(best_box), pEx, mean, stdev);
```
此函数将frame图像best_box区域的图像片归一化为均值为0的15*15大小的patch，存于pEx（用于最近邻分类器的正样本）正样本中（最近邻的box的Pattern），该正样本只有一个。
```
generator(frame, pt, warped, bbhull.size(), rng);
```
此函数属于PatchGenerator类的构造函数，用来对图像区域进行仿射变换，先RNG一个随机因子，再调用（）运算符产生一个变换后的正样本。
```
classifier.getFeatures(patch, grid[idx].sidx, fern);
```
函数得到输入的patch的特征fern（13位的二进制代码）；
```
pX.push_back(make_pair(fern, 1));   //positive ferns <features, labels=1>
```
然后标记为正样本，存入pX（用于集合分类器的正样本）正样本库；

以上的操作会循环 num_warps * good_boxes.size()即20 * 10 次，这样，pEx就有了一个正样本，而pX有了200个正样本了；

 

### meanStdDev(frame1(best_box), mean, stdev);

统计best_box的均值和标准差，var = pow(stdev.val[0],2) * 0.5;作为方差分类器的阈值。

 
### generateNegativeData(frame1);

由于TLD仅跟踪一个目标，所以我们确定了目标框了，故除目标框外的其他图像都是负样本，**无需仿射变换**；具体实现如下：

由于之前重叠度小于0.2的，都归入 bad_boxes了，所以数量挺多，把**方差大于var  0.5f的bad_boxes**（使得负样本很丰富）都加入负样本，同上面一样，需要classifier.getFeatures(patch, grid[idx].sidx, fern);和nX.push_back(make_pair(fern, 0));得到对应的fern特征和标签的nX负样本（用于集合分类器的负样本）；

然后随机在上面的bad_boxes中取bad_patches（100个）个box，然后用 getPattern函数将frame图像bad_box区域的图像片归一化到15*15大小的patch，存在nEx（用于最近邻分类器的负样本）负样本中。

这样nEx和nX都有负样本了；（**box的方差通过积分图像计算**用于方差检测器）

然后将nEx的一半作为训练集nEx，另一半作为测试集nExT；同样，nX也拆分为训练集nX和测试集nXT；

将负样本的特征nX和正样本特征pX合并到ferns_data[]中，用于集合分类器的训练；

将上面得到的一个正样本pEx和nEx合并到nn_data[]中，用于最近邻分类器的训练；

 
### 训练

这两个训练方法就是简单的 模板匹配法，集合分类器训练的是后验概率，而最近邻分类器训练的是啥

用上面的样本训练集训练集合分类器（森林） 和 最近邻分类器：
classifier.trainF(ferns_data, 2); //bootstrap = 2

对每一个样本ferns_data[i]  ，如果样本是正样本标签， 先用measure_forest函数 返回该 样本所有树的所有特征值对应的后验概率累加值，该累加值如果小于正样本阈值（0.6* nstructs，表示平均值需要大于0.6（0.6* nstructs / nstructs）,0.6是程序初始化时定的集合分类器的阈值，为经验值，后面会用测试集来评估修改，找到最优），同时用update函数更新后验概率。

classifier.trainNN(nn_data);

对每一个样本nn_data，如果标签是正样本，通过NNConf(nn_examples[i], isin, conf, dummy);计算输入图像片与在线模型之间的相关相似度conf，如果相关相似度小于0.65 ，则认为其不含有前景目标，也就是分类错误了；这时候就把它加到正样本库。然后就通过pEx.push_back(nn_examples[i]);将该样本添加到pEx正样本库中；同样，如果出现负样本分类错误，就添加到负样本库。

 
### 分类器评价 ？ 

用测试集在上面得到的 集合分类器（森林） 和 最近邻分类器中分类，评价并修改得到**最好的分类器阈值**。

  classifier.evaluateTh(nXT, nExT);

对集合分类器，对每一个测试集nXT，所有基本分类器的后验概率的平均值如果大于thr_fern（0.6），则认为含有前景目标，然后取最大的平均值（如果大于thr_fern）作为该集合分类器的新的阈值。

对最近邻分类器，对每一个测试集nExT，最大相关相似度如果大于nn_fern（0.65），则认为含有前景目标，然后取最大的最大相关相似度（如果大于nn_fern）作为该最近邻分类器的新的阈值。

## 处理视频 

```cpp
processFrame(last_gray, current_gray, pts1, pts2, pbox, status, tl, bb_file);
```
逐帧读入图片序列，进行算法处理。processFrame共包含四个模块（依次处理）：跟踪模块、检测模块、综合模块和学习模块；

## 跟踪模块

http://docs.opencv.org/3.1.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

## 其中normCrossCorrelation
其中normCrossCorrelation(img1,img2,points1,points2)是对光流法跟踪的结果不放心，因此希望通过对比前后两点周围的小块的相似性，来进一步去掉不稳定的点。这次的相似性不是相关系数，而是normalized cross-correlation (NCC)：

### 模板匹配 matchTemplate

[opencv document matchTemplate](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)

这个是opnecv中的一个函数 模板匹配是一项在一幅图像中寻找与另一幅模板图像最匹配(相似)部分的技术.


需要2幅图像:
原图像 (I): 在这幅图像里,我们希望找到一块和模板匹配的区域
模板 (T): 将和原图像比照的图像块
目标是检测最匹配的区域:


通过 滑动, 图像块一次移动一个像素 (从左往右,从上往下). 在每一个位置, 都进行一次度量计算来表明它是 “好” 或 “坏” 地与那个位置匹配 (或者说块图像和原图像的特定区域有多么相似).
对于 T 覆盖在 I 上的每个位置,你把度量值 保存 到 结果图像矩阵 (R) 中. 在 R 中的每个位置 (x,y) 都包含匹配度量值(以此点开始的块的相似度)

opencv提供的模板匹配相似度计算方法有6类

标准相关匹配 method=CV_TM_CCORR_NORMED 



# 参考资料

[TLD2010年论文Tracking-Learning-Detection](http://159.226.251.229/videoplayer/Kalal-PAMI-2011(1).pdf?ich_u_r_i=9dc166e19827cc6c08b5afa9f474c60f&ich_s_t_a_r_t=0&ich_e_n_d=0&ich_k_e_y=1745018919750763292479&ich_t_y_p_e=1&ich_d_i_s_k_i_d=4&ich_u_n_i_t=1)



[Forward-Backward Error: Automatic Detection of Tracking Failures](https://dspace.cvut.cz/bitstream/handle/10467/9553/2010-forward-backward-error-automatic-detection-of-tracking-failures.pdf?sequence=1&isAllowed=y)



[计算机视觉、机器学习相关领域论文和源代码大集合(持续更新)](http://www.voidcn.com/blog/suky520/article/p-3284.html)

[TLD源码分析](http://blog.csdn.net/zouxy09/article/details/7893011)

[庖丁解牛TLD（一）——开篇](http://blog.csdn.net/yang_xian521/article/details/6952870)


[这个讲得比较清楚 —— TLD之学习篇（四）
](http://blog.csdn.net/ttransposition/article/details/43196025)