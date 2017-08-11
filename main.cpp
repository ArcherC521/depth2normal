#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>
#include "opencv/cv.hpp"
#include "Eigen/Eigen"
#include "pcl/point_types.h"
#include "pcl/features/normal_3d.h"
#include "pcl/search/kdtree.h"
#include "pcl/io/ply_io.h"

void depth2normal(
	const Eigen::Matrix4f &I,
	const Eigen::MatrixXf &R,
	const Eigen::Vector4f &T,
	const std::string &depthMapFile,
	const std::string &modelFile,
	const std::string &modelnFile,
	const std::string &normalMapFile
) {
	cv::Mat depthMap = cv::imread(depthMapFile, cv::IMREAD_UNCHANGED);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::PointNormal>::Ptr cloudn(new pcl::PointCloud<pcl::PointNormal>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());

	for (int u = 0; u < depthMap.cols; ++u)
		for (int v = 0; v < depthMap.rows; ++v) {
			if (0 == depthMap.at<ushort>(v, u)) {
				cloud->push_back(pcl::PointXYZ(0.f, 0.f, 0.f));
				continue;
			}
			float d = static_cast<float>(depthMap.at<ushort>(v, u)) / 1000.f;
			Eigen::MatrixXf A = I * R;
			Eigen::Vector4f B = Eigen::Vector4f(d * (u + 1), d * (v + 1), d, 1.f) - I * T;
			Eigen::Vector3f C = A.colPivHouseholderQr().solve(B);
			cloud->push_back(pcl::PointXYZ(C[0], C[1], C[2]));
		}

	pcl::io::savePLYFile(modelFile, *cloud);
	ne->setInputCloud(cloud);
	ne->setSearchMethod(kdtree);
	ne->setKSearch(15);
	ne->compute(*normal);
	pcl::copyPointCloud(*cloud, *cloudn);
	pcl::copyPointCloud(*normal, *cloudn);
	pcl::io::savePLYFile(modelnFile, *cloudn);

	cv::Mat normalMap = cv::Mat::zeros(depthMap.rows, depthMap.cols, CV_32FC3);
	std::vector<cv::Mat> channels;
	cv::split(normalMap, channels);
	for (int i = 0; i < cloudn->size(); ++i) {
		Eigen::Vector4f coor = I * R * Eigen::Vector3f((*cloudn)[i].x, (*cloudn)[i].y, (*cloudn)[i].z) + I * T;
		int u = static_cast<int>(std::round(coor[0] / coor[2] - 1));
		int v = static_cast<int>(std::round(coor[1] / coor[2] - 1));
		if (0 <= u && u < depthMap.cols && 0 <= v && v < depthMap.rows) {
			channels.at(0).at<float>(v, u) = (1 - (*cloudn)[i].normal_z) / 2.f;
			channels.at(1).at<float>(v, u) = (1 - (*cloudn)[i].normal_y) / 2.f;
			channels.at(2).at<float>(v, u) = (1 - (*cloudn)[i].normal_x) / 2.f;
		}
	}
	cv::merge(channels, normalMap);
	normalMap.convertTo(normalMap, CV_8U, 255);
	cv::imwrite(normalMapFile, normalMap);
}

int main(int argc, char **argv) {
	std::string depthMapFile("../depth.png");
	std::string poseFile("../pose.txt");
	std::string modelFile("../WithoutNormal.ply");
	std::string modelnFile("../WithNormal.ply");
	std::string normalMapFile("../normalMap.png");

	float fx = 568.626f;
	float fy = 568.560f;
	float cx = 322.270f;
	float cy = 245.913f;

	try {
		Eigen::Matrix4f I;
		I <<
			 fx, 0.f,  cx, 0.f,
			0.f,  fy,  cy, 0.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f;

		Eigen::MatrixXf R(4, 3);
		Eigen::Vector4f T;
		std::ifstream is(poseFile);
		for (int i = 0; i < 4; ++i) is >> R(i, 0) >> R(i, 1) >> R(i, 2) >> T[i];

		depth2normal(I, R, T, depthMapFile, modelFile, modelnFile, normalMapFile);
	}
	catch (const std::exception &e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	return 0;
}