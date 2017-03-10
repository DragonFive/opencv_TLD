#include <LKTracker.h>
using namespace cv;

LKTracker::LKTracker(){
  term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);//迭代终止条件
  window_size = Size(4,4);  //窗口尺寸
  level = 5;				//用于设置构建的图像金字塔的栈的层数，若设置为0，则不使用金字塔。
  lambda = 0.5;
}


bool LKTracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //Forward-Backward tracking
  //基于Forward-Backward Error的中值流跟踪方法  
  //金字塔LK光流法跟踪 
  //输入： img1 前一帧图片 img2 当前帧图片 pints1 需要发现光流的点集 points2  包含新计算出来的位置的点集  winSize 表示每层金字塔的窗口大小 
  //term_criteria: 准则，指定在每个金字塔层，为某点寻找光流的迭代过程的终止条件。
/*flags
其它选项：
CV_LKFLOW_PYR_A_READY , 在调用之前，第一帧的金字塔已经准备好
CV_LKFLOW_PYR_B_READY , 在调用之前，第二帧的金字塔已经准备好
CV_LKFLOW_INITIAL_GUESSES , 在调用之前，数组 B 包含特征的初始坐标 （Hunnish: 在本节中没有出现数组 B，不知是指的哪一个）
函数 cvCalcOpticalFlowPyrLK 实现了金字塔中 Lucas-Kanade 光流计算的稀疏迭代版本 ([Bouguet00])。 
它根据给出的前一帧特征点坐标计算当前视频帧上的特征点坐标。 函数寻找具有子象素精度的坐标值。

两个参数 prev_pyr 和 curr_pyr 都遵循下列规则： 如果图像指针为 0, 函数在内部为其分配缓存空间，计算金字塔，
然后再处理过后释放缓存。 否则，函数计算金字塔且存储它到缓存中，除非设置标识 CV_LKFLOW_PYR_A[B]_READY 。 
图像应该足够大以便能够容纳 Gaussian 金字塔数据。调用函数以后，金字塔被计算而且相应图像的标识可以被设置，为下一次调用准备就绪 
(比如：对除了第一个图像的所有图像序列，标识 CV_LKFLOW_PYR_A_READY 被设置).

上面的对opencv中对这个函数的具体介绍： 我把自己使用中获得的收获写下
1 这里的pre 和 cur 分别代表我们需要跟踪图像的前一帧和当前帧，对物体的跟踪，我们在第一帧的时候一般是定位特征点，
同时把这帧保存为 pre ,那么处理第二帧的时候这两个参数都会有了。可以进行跟踪了。
2 关于 flag， 根据我们的金字搭是否建立了，设置不同的值，一般在第一次使用0，在一帧处理完了保留当前金字塔为前一帧金字塔，下次处理时候直接可以使用。
*/

  //输出： status 是bool数组 表示当前的特征点有没有被找到 similarity  表示移动点之间的差值,如果是NULL表示没有匹配上
//http://docs.opencv.org/3.1.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323  

	// forward trajectory  前向轨迹跟踪   status数组保存的是哪些点呗成功跟踪。
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
	// backward trajectory 后向轨迹跟踪
  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
  //Compute the real FB-error
    //Compute the real FB-error  
  //原理很简单：从t时刻的图像的A点，跟踪到t+1时刻的图像B点；然后倒回来，从t+1时刻的图像的B点往回跟踪，  
  //假如跟踪到t时刻的图像的C点，这样就产生了前向和后向两个轨迹，比较t时刻中 A点 和 C点 的距离，如果距离  
  //小于一个阈值，那么就认为前向跟踪是正确的；这个距离就是FB_error  
  //计算 前向 与 后向 轨迹的误差  
  for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);//残差为欧式距离 
  }
  //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
  normCrossCorrelation(img1,img2,points1,points2);
  return filterPts(points1,points2);
}
//利用NCC把跟踪预测的结果周围取10*10的小图片与原始位置周围10*10的小图片
//（使用函数getRectSubPix得到）进行模板匹配（调用matchTemplate）  
void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);

        for (int i = 0; i < points1.size(); i++) {
			//为1表示该特征点跟踪成功  
            //从前一帧和当前帧图像中（以每个特征点为中心？）提取10x10象素矩形，使用亚象素精度 
                if (status[i] == 1) {
                        getRectSubPix( img1, Size(10,10), points1[i],rec0 );//前一帧的位置上;
                        getRectSubPix( img2, Size(10,10), points2[i],rec1);//后一帧的位置上;
						////匹配前一帧和当前帧中提取的10x10象素矩形，得到匹配后的映射图像  
                        //CV_TM_CCOEFF_NORMED 归一化相关系数匹配法  
                        //参数分别为：欲搜索的图像。搜索模板。比较结果的映射图像。指定匹配方法  所以是在1模板上找有没有图像0;
                        matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
                        similarity[i] = ((float *)(res.data))[0];//得到各个特征的相似度大小？

                } else {
                        similarity[i] = 0.0;
                }
        }
        rec0.release();
        rec1.release(); 
        res.release();
}

//筛选出 FB_error[i] <= median(FB_error) 和 sim_error[i] > median(sim_error) 的特征点  
//得到NCC和FB error结果的中值，分别去掉中值一半的跟踪结果不好的点  
bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  simmed = median(similarity);			//找到相似度的中值;
  size_t i, k;						
  for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
          continue;
        if(similarity[i]> simmed){		//剩下 similarity[i]> simmed 的特征点  
          points1[k] = points1[i];
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)
    return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);
//再对上一步剩下的特征点进一步筛选，剩下 FB_error[i] <= fbmed 的特征点   但这里没有筛选FB_error
  fbmed = median(FB_error);
  for( i=k = 0; i<points2.size(); ++i ){
      if( !status[i])
        continue;
      if(FB_error[i] <= fbmed){
        points1[k] = points1[i];
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0) //表示如果能有点通过测试 那么就跟踪对了；
    return true;
  else
    return false;
}




/*
 * old OpenCV style
void LKTracker::init(Mat img0, vector<Point2f> &points){
  //Preallocate
  //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //const int NUM_PTS = points.size();
  //status = new char[NUM_PTS];
  //track_error = new float[NUM_PTS];
  //FB_error = new float[NUM_PTS];
}


void LKTracker::trackf2f(..){
  cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
  cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
}
*/

