#include "stdio.h"
#include<iostream> 
#include <string>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>


using namespace cv;
using namespace std;
string g_str_src;
string g_str_bg;
string g_str_face;
double g_thresh = 100.0;

CascadeClassifier faceCascade;
CascadeClassifier eyes_Cascade;

void RemoveBlackBorder(Mat& iplImg, Mat& dstImg);
Mat RemoveBlackCorner(Mat img);
bool parse_cmd(int argc, char* argv[])
{
	if (argc < 3)
	{
		return false;
	}

	g_str_src = string(argv[1]);
	g_str_bg = string(argv[2]);

	if (argc > 3)
	{
		g_thresh = stold(argv[3]);
	}

	return true;
}

string GetFolderFromFile(string strFile)
{
	size_t last_slash = strFile.find_last_of("\\");
	std::string directory = strFile.substr(0, last_slash);
	return directory;
}


int DetectFace(Mat img, Mat imgGray) {
	namedWindow("src", WINDOW_AUTOSIZE);
	vector<Rect> faces, eyes;
	faceCascade.detectMultiScale(imgGray, faces, 1.2, 5, 0, Size(30, 30));
	int retVal = -1;
	//目前只取一个脸
	if (faces.size() > 0) {
		for (size_t i = 0; i < faces.size(); i++) {
			//框出人脸位置
			rectangle(img, Point(faces[i].x+ faces[i].width / 8, faces[i].y+faces[i].height / 8), 
				Point(faces[i].x + faces[i].width*7/8, faces[i].y + faces[i].height * 7 / 8), Scalar(0, 0, 255), 1, 8);
			cout << faces[i] << endl;
			//将人脸从灰度图中抠出来
			Mat face_ = imgGray(faces[i]);
			//缩小一点，默认取的矩形比较大
			Rect rect(Point(faces[i].width / 8, faces[i].height / 8),
				Point(faces[i].width * 7 / 8,  faces[i].height * 7/ 8));
			Mat ROI = face_(rect);
			//RemoveBlackBorder(ROI, ROI);
			Mat imgOut = RemoveBlackCorner(ROI);
			//RemoveBlackBorder(ROI, imgOut);
			imwrite(g_str_face, imgOut);
			retVal = 0;
			eyes_Cascade.detectMultiScale(face_, eyes, 1.2, 2, 0, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++) {
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
				circle(img, eye_center, radius, Scalar(65, 105, 255), 4, 8, 0);
			}
		}
	}
	imshow("src", img);
	return retVal;
}

int InitFaceDetect()
{
	if (!faceCascade.load("D:\\Workspace\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")) {
		cout << "人脸检测级联分类器没找到！！" << endl;
		return -1;
	}
	if (!eyes_Cascade.load("D:\\Workspace\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml")) {
		cout << "眼睛检测级联分类器没找到！！" << endl;
		return -1;
	}

	return 0;
}

/************************************************************************/
/* 消除图片四周的黑色边框                                               */
/************************************************************************/
void RemoveBlackBorder(Mat& iplImg, Mat& dstImg)
{
	int width = iplImg.size().width;
	int height = iplImg.size().height;
	int a = 0, b = 0, c = 0, d = 0;
	int i = 0, j = 0;
	string strImg;
	//IplImage* img = cvLoadImage(strImg, 0);
	//cvvImage img()

	if (iplImg.channels() == 1)	//灰度图片
	{
		//消除黑色边框：上
		for (j = 0; j < height; j++)
		{
			bool flag = false;
			for (i = 0; i < width; i++)
			{
				if (iplImg.at<uchar>(j, i) < 30)
				{
					;
				}
				else
				{
					flag = true;
					a = j;
					break;
				}
			}
			if (flag) break;
		}
		//消除黑色边框：下
		for (j = height - 1; j >= a; j--)
		{
			bool flag = false;
			for (i = 0; i < width; i++)
			{
				if (iplImg.at<uchar>(j, i) < 30)
				{
					;
				}
				else
				{
					flag = true;
					b = j;
					break;
				}
			}
			if (flag) break;
		}
		//消除黑色边框：左
		for (i = 0; i < width; i++)
		{
			bool flag = false;
			for (j = 0; j < height; j++)
			{
				if (iplImg.at<uchar>(j, i) < 30)
				{
					;
				}
				else
				{
					flag = true;
					c = i;
					break;
				}
			}
			if (flag) break;
		}
		//消除黑色边框：右
		for (i = width - 1; i >= c; i--)
		{
			bool flag = false;
			for (j = 0; j < height; j++)
			{
				if (iplImg.at<uchar>(j, i) < 30)
				{
					;
				}
				else
				{
					flag = true;
					d = i;
					break;
				}
			}
			if (flag) break;
		}
	}
	else if (iplImg.channels() == 3)	//彩色图片
	{
		//消除黑色边框：上
		for (j = 0; j < height; j++)
		{
			bool flag = false;
			for (i = 0; i < width; i++)
			{
				int tmpb, tmpg, tmpr;
				tmpb = iplImg.at<Vec3b>(j, i)[2];
				tmpg = iplImg.at<Vec3b>(j, i)[1];
				tmpr = iplImg.at<Vec3b>(j, i)[0];
				if (tmpb <= 30 && tmpg <= 30 && tmpr <= 30)
				{
					;
				}
				else
				{
					flag = true;
					a = j;
					break;
				}
			}
			if (flag) break;
		}
		//printf("上 a: %d\n", a);

		//消除黑色边框：下
		for (j = height - 1; j >= a; j--)
		{
			bool flag = false;
			for (i = 0; i < width; i++)
			{
				int tmpb, tmpg, tmpr;
				tmpb = iplImg.at<Vec3b>(j, i)[2];
				tmpg = iplImg.at<Vec3b>(j, i)[1];
				tmpr = iplImg.at<Vec3b>(j, i)[0];
				if (tmpb <= 30 && tmpg <= 30 && tmpr <= 30)
				{
					;
				}
				else
				{
					flag = true;
					b = j;
					break;
				}
			}
			if (flag) break;
		}
		//printf("下 b: %d\n", b);

		//消除黑色边框：左
		for (i = 0; i < width; i++)
		{
			bool flag = false;
			for (j = 0; j < height; j++)
			{
				int tmpb, tmpg, tmpr;
				tmpb = iplImg.at<Vec3b>(j, i)[2];
				tmpg = iplImg.at<Vec3b>(j, i)[1];
				tmpr = iplImg.at<Vec3b>(j, i)[0];
				if (tmpb <= 30 && tmpg <= 30 && tmpr <= 30)
				{
					;
				}
				else
				{
					flag = true;
					c = i;
					break;
				}
			}
			if (flag) break;
		}
		//printf("左 c: %d\n", c);

		//消除黑色边框：右
		for (i = width - 1; i >= c; i--)
		{
			bool flag = false;
			for (j = 0; j < height; j++)
			{
				int tmpb, tmpg, tmpr;
				tmpb = iplImg.at<Vec3b>(j, i)[2];
				tmpg = iplImg.at<Vec3b>(j, i)[1];
				tmpr = iplImg.at<Vec3b>(j, i)[0];
				if (tmpb <= 30 && tmpg <= 30 && tmpr <= 30)
				{
					;
				}
				else
				{
					flag = true;
					d = i;
					break;
				}
			}
			if (flag) break;
		}
		//printf("右 d: %d\n", d);
	}

	//复制图像
	int w = d - c + 1, h = b - a + 1;
	dstImg = Mat(iplImg, Rect(c, a, w, h));

	return;
}

/************************************************************************/
/* 消除图片四周的黑色边角区域                                           */
/************************************************************************/

Mat RemoveBlackCorner(Mat img)
{
	int i, j;
	int h = img.size().height;
	int w = img.size().width;

	if (img.channels() == 1)	//灰度图片
	{
		for (j = 0; j < h; j++)
		{
			for (i = 0; i < w; i++)
			{
				if (img.at<uchar>(j, i) < 110)
				{
					img.at<uchar>(j, i) = 255;
				}
				else
				{
					break;
				}
			}
			for (i = w - 1; i >= 0; i--)
			{
				if (img.at<uchar>(j, i) < 110)
				{
					img.at<uchar>(j, i) = 255;
				}
				else
				{
					break;
				}
			}
		}
		for (i = 0; i < w; i++)
		{
			for (j = 0; j < h; j++)
			{
				if (img.at<uchar>(j, i) < 110)
				{
					img.at<uchar>(j, i) = 255;
				}
				else
				{
					break;
				}
			}
			for (j = h - 1; j >= 0; j--)
			{
				if (img.at<uchar>(j, i) < 110)
				{
					img.at<uchar>(j, i) = 255;
				}
				else
				{
					break;
				}
			}
		}
	}

	return img;
}



int main(int argc, char* argv[])
{
	if (!parse_cmd(argc, argv))
	{
		cout << "command error" << endl;
		return -1;
	}
	
	if (InitFaceDetect() != 0)
	{
		return -1;
	}
	
	//
	string strDirBase = GetFolderFromFile(g_str_src);
	Mat img_src = imread(g_str_src);
	Mat img_background = imread(g_str_bg);
	g_str_face = strDirBase + "\\tmp_face.jpg";
#ifdef _DBG_SHOW
	namedWindow("img_src");
	imshow("img_src", img_src);
#endif

	Mat img_gray;
	cvtColor(img_src, img_gray, COLOR_BGR2GRAY); //图像灰度化

	int nFace = DetectFace(img_src, img_gray);
	waitKey(3000);
#ifdef _DBG_SHOW
	namedWindow("gray", WINDOW_NORMAL);
	imshow("gray", img_gray);
#endif
	// 按照背景图大小等比缩放
	Size dsize = Size(img_background.cols * 0.55, img_background.rows * 0.55);
	//判断一下是否自动检测到了人脸
	Mat img_face;
	if (nFace == 0) 
	{
		cout << "opencv find face,get face." << endl;
		img_face = imread(g_str_face);
	}
	else
	{
		cout << "can not find face.use image user input." << endl;
		img_face = img_gray;
	}
	resize(img_face, img_face, dsize, 1, 1, INTER_AREA);

	//输出缩放后效果图并重新加载
	cout << "g_thresh = " << g_thresh << endl;
	Mat img_face2;
	threshold(img_face, img_face2, g_thresh, 255, THRESH_BINARY);
	imwrite(strDirBase + "\\tmp.jpg", img_face2);
	//imshow("img_face2", img_face2);
	Mat img_face3 = imread(strDirBase + "\\tmp.jpg");
	//居中粘合两图
	Rect roi_rect = Rect((img_background.cols - img_face3.cols) / 2, (img_background.rows - img_face3.rows) / 2 -13
		, img_face3.cols, img_face3.rows);
	img_face3.copyTo(img_background(roi_rect));
	//显示并输出
	imshow("mixed", img_background);

	imwrite(g_str_src + ".emoji.jpg", img_background);

	waitKey(5000);

	destroyAllWindows();
	return 0;
}
