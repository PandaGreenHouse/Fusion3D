#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/pcl_config.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/pca.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/io/ply_io.h>
#include "opencv_2010\opencv2\core.hpp"
#include "opencv_2010\opencv2\highgui\highgui.hpp"
#include "opencv_2010\opencv2\imgproc.hpp"

bool intersectTri(Eigen::Vector3f &v0, Eigen::Vector3f &v1, Eigen::Vector3f &v2,
	Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, float &u, float &v, float &dist, Eigen::Vector3f &interPos);
void setViewerPose(pcl::visualization::PCLVisualizer::Ptr viewer, Eigen::Affine3f &pose);
bool getIntersection(Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, pcl::PolygonMesh &mesh, Eigen::Vector3f &intersection);
bool pickingMesh(int px, int py, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PolygonMesh &mesh,
				 pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_picked);
bool pickingMesh(Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, pcl::PolygonMesh &mesh, pcl::PointXYZRGB& point_picked);
void setViewerPose(pcl::visualization::PCLVisualizer::Ptr viewer, Eigen::Affine3f &pose);
void setViewerFromMesh(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PolygonMesh &mesh);
void drawPickingsOnImg(cv::Mat& img, int px, int py);
void decimateMesh(pcl::PolygonMesh& mesh_in, pcl::PolygonMesh& mesh_out);