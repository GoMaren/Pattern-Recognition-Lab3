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

int** run(const int height, const int width, const int modKs, int Ks[], int*** q, double** g, const int loops)
{
	// Initialize phi, L, R, U, D
	int modT = height * width;
	double** phi = new double* [modT];
	double** L = new double* [modT];
	double** R = new double* [modT];
	double** U = new double* [modT];
	double** D = new double* [modT];
	for (int ij = 0; ij < modT; ++ij)
	{
		phi[ij] = new double[modKs]();
		L[ij] = new double[modKs]();
		R[ij] = new double[modKs]();
		U[ij] = new double[modKs]();
		D[ij] = new double[modKs]();
	}

	for (int i = height - 2; i >= 0; --i)
	{
		const int i_ = i * width;
		for (int j = width - 2; j >= 0; --j)
		{
			const int ij = i_ + j;
			for (int k = 0; k < modKs; ++k)
			{
				double maxR = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g[k_][k];
					if (R_ > maxR)
						maxR = R_;
				}
				R[i_ + j][k] = maxR;

				double maxD = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g[k_][k];
					if (D_ > maxD)
						maxD = D_;
				}
				D[i_ + j][k] = maxD;
			}
		}
	}

	auto start = high_resolution_clock::now();
	// Main loop
	for (int iter = 0; iter < loops; ++iter)
	{
		// Forward
		for (int i = 1; i < height; ++i)
		{
			const int i_ = i * width;
			for (int j = 1; j < width; ++j)
			{
				const int ij = i_ + j;
				for (int k = 0; k < modKs; ++k)
				{
					double maxL = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double L_ = L[ij - 1][k_] + 0.5 * q[i][j - 1][k_] + g[k_][k] - phi[ij - 1][k_];
						if (L_ > maxL)
							maxL = L_;
					}
					L[ij][k] = maxL;

					double maxU = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double U_ = U[ij - width][k_] + 0.5 * q[i - 1][j][k_] + g[k_][k] + phi[ij - width][k_];
						if (U_ > maxU)
							maxU = U_;
					}
					U[ij][k] = maxU;

					phi[ij][k] = (L[ij][k] + R[ij][k] - U[ij][k] - D[ij][k]) * 0.5;
				}
			}
		}
		// Backward
		for (int i = height - 2; i >= 0; --i)
		{
			const int i_ = i * width;
			for (int j = width - 2; j >= 0; --j)
			{
				const int ij = i_ + j;
				for (int k = 0; k < modKs; ++k)
				{
					double maxR = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g[k_][k] - phi[ij + 1][k_];
						if (R_ > maxR)
							maxR = R_;
					}
					R[ij][k] = maxR;

					double maxD = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g[k_][k] + phi[ij + width][k_];
						if (D_ > maxD)
							maxD = D_;
					}
					D[ij][k] = maxD;

					phi[ij][k] = (L[ij][k] + R[ij][k] - U[ij][k] - D[ij][k]) * 0.5;
				}
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time used for " << loops << " iterations : " << double(duration.count()) / 1000000. << endl;

	// Best Ks
	int** res = new int* [height];
	for (int i = 0; i < height; ++i)
	{
		res[i] = new int[width]();
	}

	for (int i = 0; i < height; ++i)
	{
		const int i_ = i * width;
		for (int j = 0; j < width; ++j)
		{
			const int ij = i_ + j;
			int k_star = 0;
			double value = -10000000.;
			for (int k_ = 0; k_ < modKs; ++k_)
			{
				const double v_ = L[ij][k_] + R[ij][k_] + 0.5 * q[i][j][k_] - phi[ij][k_];
				if (v_ > value)
				{
					value = v_;
					k_star = k_;
				}
			}
			res[i][j] = Ks[k_star];
		}
	}

	return res;
}

int main()
{
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

	const int loops = 10;
	int** res = run(height, width, modKs, Ks, q, g, loops);

	Mat result = Mat::zeros(Size(width, height), CV_8UC1);
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			result.at<uchar>(i, j) = uchar(res[i][j]);

	namedWindow("Result image", WINDOW_AUTOSIZE);
	imshow("Result image", result);
	imwrite("res1.png", result);

	waitKey(0);
	return 0;
}