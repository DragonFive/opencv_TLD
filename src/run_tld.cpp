#include <opencv2/opencv.hpp>
#include <tld_utils.h>
#include <iostream>
#include <sstream>
#include <TLD.h>
#include <stdio.h>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

void readBB(char* file){	//读取每行字符串并挨个转化成数字；
  ifstream bb_file (file);
  string line;
  getline(bb_file,line);
  istringstream linestream(line);
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
  for (int i=0;i<argc;i++){
      if (strcmp(argv[i],"-b")==0){		//bounding_boxes 文件;
          if (argc>i){
              readBB(argv[i+1]);
              gotBB = true;
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-s")==0){		//表示读入视频的位置;
          if (argc>i){
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }
      if (strcmp(argv[i],"-p")==0){		//初始化的一系列参数;
          if (argc>i){
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0){
          tl = false;
      }
      if (strcmp(argv[i],"-r")==0){
          rep = true;
      }
  }
}

int main(int argc, char * argv[]){
  VideoCapture capture;
  capture.open(0);
  //opencv提供的FileStorage类。可以保存浮点数据或XML/YML文件 
  FileStorage fs;
  //Read options
  read_options(argc,argv,capture,fs);
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback( "TLD", mouseHandler, NULL );
  //TLD framework
  TLD tld;
  //Read parameters file 从配置文件里面读取初始化配置
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  
/***********下面从文件或者摄像头读取视频************/
  if (fromfile){//从文件中读取数据
      capture >> frame;
      cvtColor(frame, last_gray, CV_RGB2GRAY);
      frame.copyTo(first);
  }else{// //如果为读取摄像头，则设置获取的图像大小为320x240 
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }
/***********上面从文件或者摄像头读取视频************/

/***********下面获取boundingbox**************/
  ///Initialization  
GETBOUNDINGBOX:
  while(!gotBB)//如果没有指定初始化bounding box 的位置 就需要手工画出bounding box，但我认为这里程序的流程使用goto之后 是有问题的。 
  {
    if (!fromfile){
      capture >> frame;
    }
    else//不明白这一步 为什么重复来操作呢
      first.copyTo(frame);
    cvtColor(frame, last_gray, CV_RGB2GRAY);
    drawBox(frame,box);//手动画出boundingbox
    imshow("TLD", frame);
    if (cvWaitKey(33) == 'q')
	    return 0;
  }
  //由于图像片（min_win 为15x15像素）是在bounding box中采样得到的，所以box必须比min_win要大  
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  /***********上面获取boundingbox**************/
  
  //Remove callback
  //如果上面的程序能够流畅第执行下来，那么久认为已经获得第一帧用户框定的box了，就取消鼠标响应
  cvSetMouseCallback( "TLD", NULL, NULL );
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  
  
  //TLD initialization TLD的初始化工作
  tld.init(last_gray,box,bb_file);

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;		//记录跟踪成功与否的状态 lastbox been found  
  int frames = 1;		//记录已过去帧数
  int detections = 1;  	//记录成功检测到的目标box数目   
REPEAT:
  while(capture.read(frame)){
    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);
    //Process Frame
    tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
    //Draw Points
    if (status){
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0));	////当前的特征点用绿色点表示  
      drawBox(frame,pbox);
      detections++;
    }
    //Display
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray,current_gray);
    pts1.clear();
    pts2.clear();
    frames++;
    printf("Detection rate: %d/%d\n",detections,frames);
    if (cvWaitKey(33) == 'q')
      break;
  }
  if (rep){
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;
}
