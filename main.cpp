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

void depth2normal(
	const Eigen::Matrix4f &I,
	const Eigen::MatrixXf &R,
	const Eigen::Vector4f &T,
	const std::string &depthMapFile,
	const std::string &normalMapFile
) {
	cv::Mat depthMap = cv::imread(depthMapFile, cv::IMREAD_UNCHANGED);

	std::vector<std::vector<Eigen::Vector3f>> coor(depthMap.rows);
	for (int row = 0; row < depthMap.rows; ++row) coor[row].reserve(depthMap.cols);
	for (int u = 0; u < depthMap.cols; ++u)
		for (int v = 0; v < depthMap.rows; ++v) {
			if (0 == depthMap.at<ushort>(v, u)) {
				coor[v][u] = Eigen::Vector3f(0.f, 0.f, 0.f);
				continue;
			}
			float d = static_cast<float>(depthMap.at<ushort>(v, u)) / 1000.f;
			Eigen::MatrixXf A = I * R;
			Eigen::Vector4f B = Eigen::Vector4f(d * (u + 1), d * (v + 1), d, 1.f) - I * T;
			coor[v][u] = A.colPivHouseholderQr().solve(B);
		}

	cv::Mat normalMap = cv::Mat::zeros(depthMap.rows, depthMap.cols, CV_32FC3);
	for (int u = 0; u < depthMap.cols; ++u)
		for (int v = 0; v < depthMap.rows; ++v) {
			Eigen::Vector3f normal;

			Eigen::Vector3f vecUp, vecDown, vecLeft, vecRight;
			if (0 < v) vecUp = coor[v - 1][u] - coor[v][u];
			if (v < depthMap.rows - 1) vecDown = coor[v + 1][u] - coor[v][u];
			if (0 < u) vecLeft = coor[v][u - 1] - coor[v][u];
			if (u < depthMap.cols - 1) vecRight = coor[v][u + 1] - coor[v][u];

			if (0 == u && 0 == v) normal = vecDown.cross(vecRight).normalized();
			else if (0 == u && depthMap.rows - 1 == v) normal = vecRight.cross(vecUp).normalized();
			else if (depthMap.cols - 1 == u && 0 == v) normal = vecLeft.cross(vecDown).normalized();
			else if (depthMap.cols - 1 == u && depthMap.rows - 1 == v) normal = vecUp.cross(vecLeft).normalized();
			else if (0 == u) normal = (vecDown.cross(vecRight).normalized() + vecRight.cross(vecUp).normalized()).normalized();
			else if (depthMap.cols - 1 == u) normal = (vecUp.cross(vecLeft).normalized() + vecLeft.cross(vecDown).normalized()).normalized();
			else if (0 == v) normal = (vecLeft.cross(vecDown).normalized() + vecDown.cross(vecRight).normalized()).normalized();
			else if (depthMap.rows - 1 == v) normal = (vecRight.cross(vecUp).normalized() + vecUp.cross(vecLeft).normalized()).normalized();
			else normal = (vecUp.cross(vecLeft).normalized() + vecDown.cross(vecRight).normalized()).normalized();
			
			normalMap.at<cv::Vec3f>(v, u) = cv::Vec3f((1 - normal[2]) / 2.f, (1 - normal[1]) / 2.f, (1 - normal[0]) / 2.f);
		}
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

		// depth2normal(I, R, T, depthMapFile, modelFile, modelnFile, normalMapFile);
		depth2normal(I, R, T, depthMapFile, normalMapFile);
	}
	catch (const std::exception &e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	return 0;
}
