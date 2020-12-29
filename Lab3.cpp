#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <chrono> 
#include <cmath>
#include <cassert>

using namespace cv;
using namespace std;
using namespace std::chrono;

int qFunc(int c1, const int epsilon, int c2)
{
	int diff = int(c1 != epsilon);
	diff *= abs(c1 - c2);
	return -diff;
}

int main()
{
	// Test
	test();

	Mat image_, image;
	image_ = imread("1.png", IMREAD_UNCHANGED);
	namedWindow("Original image", WINDOW_AUTOSIZE);
	imshow("Original image", image_);
	cvtColor(image_, image, COLOR_BGR2GRAY);
	imshow("Gray image", image);
	const int height = image.size().height;
	const int width = image.size().width;

	// Get array from Mat
	int** colors = new int* [height];
	for (int i = 0; i < height; ++i)
	{
		colors[i] = new int[width];
		for (int j = 0; j < width; ++j)
		{
			colors[i][j] = int(image.at<uchar>(i, j));
		}
	}

	// Target colors
	const int epsilon = 0;
	const int modKs = 5;
	int Ks[modKs] = {0, 64, 128, 192, 255};

	// G
	const double alpha = 0.03;
	double** g = new double* [modKs];
	for (int k = 0; k < modKs; ++k)
		g[k] = new double[modKs]();

	for (int k = 0; k < modKs; ++k)
		for (int k_ = 0; k_ < modKs; ++k_)
			g[k][k_] = -alpha * abs(Ks[k] - Ks[k_]);

	// Q
	int*** q = new int** [height];
	for (int i = 0; i < height; ++i)
	{
		q[i] = new int* [width];
		for (int j = 0; j < width; ++j)
		{
			q[i][j] = new int[3];
			for (int k = 0; k < modKs; ++k)
				q[i][j][k] = qFunc(colors[i][j], epsilon, Ks[k]);
		}
	}

	waitKey(0);
	return 0;
}