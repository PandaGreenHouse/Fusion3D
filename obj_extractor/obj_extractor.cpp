// obj_extractor.cpp : Defines the exported functions for the DLL application.
//
#include "extractor_api.h"
#include "util.h"
#include "stdafx.h"

pcl::visualization::PCLVisualizer::Ptr viewer;
pcl::PolygonMesh mesh, mesh_decimated;
Eigen::Affine3f camPose;
Eigen::Matrix4f world_to_cam;
Eigen::Matrix3f calib_mat, inverse_calib_mat;
float _focal_len;
std::vector<std::string> img_paths;
std::vector<Eigen::Affine3f> camPoses;
std::vector<Eigen::Matrix4f> world_to_cams;
std::vector<Eigen::Matrix3f> calib_mats, inverse_calib_mats;
std::vector<float> focal_lens;
bool loop = true;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_picked = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
float sample_len = 0.0f;
std::string out_path = "";
float scale = 1.0f;
int n_views = 0;
int cur_view = 0;
cv::Mat img_cur;
float img_origin_w;
float img_origin_h;
std::string file_name_decimated;

float getScale()
{
	return scale;
}

void decimatingMesh()
{
	decimateMesh(mesh, mesh_decimated);
	pcl::io::savePLYFile(file_name_decimated, mesh_decimated);
}

bool pickingMeshByImage(float x, float y, Eigen::Matrix3f& inv_calib_mat, Eigen::Affine3f& pose, pcl::PolygonMesh& mesh, pcl::PointXYZRGB& point_picked)
{
	Eigen::Vector3f p;
	p[0] = x; 
	p[1] = y; 
	p[2] = 1.0f;
	Eigen::Vector3f vRayPos = pose.translation();
	Eigen::Vector3f vRayDir;
	vRayDir = inv_calib_mat*p;
	vRayDir = pose.linear()*vRayDir;
	bool bPicked = pickingMesh(vRayPos, vRayDir, mesh, point_picked);
	return bPicked;
}

void onMouse(int mouse_event, int x, int y, int e, void* f)
{
	float img_width, img_height;
	img_width = img_cur.cols;
	img_height = img_cur.rows;
	float px, py;
	float sx = img_origin_w / img_width;
	float sy = img_origin_h / img_height;
	pcl::PointXYZRGB pointPicked;
	switch (mouse_event)
	{
	case cv::EVENT_LBUTTONDOWN:
		drawPickingsOnImg(img_cur, x, y);
		px = sx*(float)x;
		py = sy*(float)y;
		if (pickingMeshByImage(px, py, inverse_calib_mat, camPose, mesh, pointPicked))
		{
			if (points_picked->size() == 2)
				points_picked->resize(0);
			points_picked->push_back(pointPicked);
		}
		break;
	case cv::EVENT_LBUTTONUP:
		break;
	}
}

void setting(float len, std::string& raw_file_name, std::string& decimate_file_name)
{
	sample_len = len;
	out_path = raw_file_name;
	file_name_decimated = decimate_file_name;
}

void scalingMesh(float s)
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud);
	for (int i = 0; i < cloud.size(); ++i)
	{
		cloud.points[i].x *= s;
		cloud.points[i].y *= s;
		cloud.points[i].z *= s;
	}
	pcl::toPCLPointCloud2(cloud, mesh.cloud);
}

int scalingMesh()
{
	if (sample_len == 0.0f)
		return 0;
	if (points_picked->points.size() < 2)
		return 1;
	Eigen::Vector3f vec = points_picked->points[1].getVector3fMap() - points_picked->points[0].getVector3fMap();
	if (vec.norm() == 0.0f)
		return 2;
	float s = sample_len / vec.norm();
	scale = s;
	for (int i = 0; i < points_picked->points.size(); ++i)
	{
		points_picked->points[i].x *= s;
		points_picked->points[i].y *= s;
		points_picked->points[i].z *= s;
	}
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud);
	for (int i = 0; i < cloud.size(); ++i)
	{
		cloud.points[i].x *= s;
		cloud.points[i].y *= s;
		cloud.points[i].z *= s;
	}
	pcl::toPCLPointCloud2(cloud, mesh.cloud);
	return 3;
}

int saveMesh()
{
	if (out_path == "")
		return 0;
	pcl::io::savePLYFile(out_path, mesh);
	return 1;
}

void addImgPath(std::string& img_path)
{
	img_paths.push_back(img_path);
}

void getCamParams(float* position, float* rot, float* worldToCam, float* calib, float* inverse_calib, float focalLen)
{
	Eigen::Vector3f vPos;
	Eigen::Matrix3f rotMat;
	vPos[0] = position[0]; vPos[1] = position[1]; vPos[2] = position[2];
	rotMat << rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8];
	camPose.linear() = rotMat;
	camPose.translation() = vPos;
	world_to_cam << worldToCam[0], worldToCam[1], worldToCam[2], worldToCam[3],
		worldToCam[4], worldToCam[5], worldToCam[6], worldToCam[7],
		worldToCam[8], worldToCam[9], worldToCam[10], worldToCam[11],
		worldToCam[12], worldToCam[13], worldToCam[14], worldToCam[15];
	calib_mat << calib[0], calib[1], calib[2],
		calib[3], calib[4], calib[5],
		calib[6], calib[7], calib[8];
	inverse_calib_mat << inverse_calib[0], inverse_calib[1], inverse_calib[2],
		inverse_calib[3], inverse_calib[4], inverse_calib[5],
		inverse_calib[6], inverse_calib[7], inverse_calib[8];
	_focal_len = focalLen;
	camPoses.push_back(camPose);
	world_to_cams.push_back(world_to_cam);
	calib_mats.push_back(calib_mat);
	inverse_calib_mats.push_back(inverse_calib_mat);
	++n_views;
}

void extractObj(std::vector<std::vector<float>>& vertices, float plane_tolerance, float cluster_tolerance)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
	cloud_in->clear();
	for (int i = 0; i < vertices.size(); ++i)
	{
		pcl::PointXYZ p;
		p.x = vertices[i][0];
		p.y = vertices[i][1];
		p.z = vertices[i][2];
		cloud_in->points.push_back(p);
	}
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(200);
	//float plane_tolerance = 0.05f;
	seg.setDistanceThreshold(plane_tolerance);//(0.05);
	seg.setInputCloud(cloud_in);
	seg.segment(*inliers, *coefficients);
	/*Eigen::Vector3f normal;
	normal[0] = coefficients->values[0];
	normal[1] = coefficients->values[1];
	normal[2] = coefficients->values[2];
	normal.normalize();
	Eigen::Hyperplane<float, 3> plane(normal, coefficients->values[3]);*/
	std::cout << "plane inlier count:" << inliers->indices.size() << std::endl;
	std::vector<int> labels;
	labels.resize(cloud_in->points.size());
	for (size_t i = 0; i < cloud_in->points.size(); ++i)
	{
		labels[i] = 1;
	}
	for (size_t i = 0; i < inliers->indices.size(); ++i)
	{
		labels[inliers->indices[i]] = 0;
	}
	vertices.clear();
	int out_size = cloud_in->points.size() - inliers->indices.size();
	vertices.resize(out_size);
	size_t count = 0;
	for (size_t i = 0; i < labels.size(); ++i)
	{
		if (labels[i] == 0)
			continue;
		vertices[count].resize(3);
		vertices[count][0] = cloud_in->points[i].x;
		vertices[count][1] = cloud_in->points[i].y;
		vertices[count][2] = cloud_in->points[i].z;
		++count;
	}
}

void extractObj(std::string& inputPlyFileName, std::string& outPlyFileName, std::vector<float>& cam_positions, 
			    float plane_tolerance, float cluster_tolerance)
{
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_clipped(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::io::loadPLYFile(inputPlyFileName, *cloud_original);
	Eigen::Vector3f vCenter(0.0f, 0.0f, 0.0f);
	int cam_count = cam_positions.size() / 3;
	std::vector<Eigen::Vector3f> cam_posVecs;
	for (size_t i = 0; i < cam_count; ++i)
	{
		Eigen::Vector3f v;
		v[0] = cam_positions[3*i];
		v[1] = cam_positions[3*i + 1];
		v[2] = cam_positions[3*i + 2];
		vCenter = (i*vCenter + v) / (i + 1);
		cam_posVecs.push_back(v);
		std::cout << "ex dll: cam position:" << cam_posVecs[i][0] << "," << cam_posVecs[i][1] << "," << cam_posVecs[i][2] << std::endl;
	}
	float fRadius = 0.0f;
	for (size_t i = 0; i < cam_posVecs.size(); ++i)
	{
		Eigen::Vector3f vec = cam_posVecs[i] - vCenter;
		if (fRadius < vec.norm())
		{
			fRadius = vec.norm();
		}
	}
	for (size_t i = 0; i < cloud_original->points.size(); ++i)
	{
		Eigen::Vector3f vec = cloud_original->points[i].getVector3fMap() - vCenter;
		if (vec.norm() > fRadius)
			continue;
		cloud_clipped->points.push_back(cloud_original->points[i]);
	}
	std::vector<int> labels;
	labels.resize(cloud_clipped->points.size());
	for (size_t i = 0; i < labels.size(); ++i)
	{
		labels[i] = 1;
	}
	if (plane_tolerance > 0.0f)
	{
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(200);
		//float plane_tolerance = 0.05f;
		seg.setDistanceThreshold(plane_tolerance);//(0.05);
		seg.setInputCloud(cloud_clipped);
		seg.segment(*inliers, *coefficients);
		for (size_t i = 0; i < inliers->indices.size(); ++i)
		{
			labels[inliers->indices[i]] = 0;
		}
	}
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_obj(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	cloud_obj->points.clear();
	for (size_t i = 0; i < labels.size(); ++i)
	{
		if (labels[i] == 0)
			continue;
		cloud_obj->points.push_back(cloud_clipped->points[i]);
	}
	pcl::io::savePLYFile(outPlyFileName, *cloud_obj);
}

void segmentPlane(std::vector<float>& points, float plane_tolerance, std::vector<int>& labels)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
	cloud_in->points.clear();
	size_t vert_num = points.size() / 3;
	for (size_t i = 0; i < vert_num; ++i)
	{
		pcl::PointXYZ p;
		p.x = points[3 * i];
		p.y = points[3 * i + 1];
		p.z = points[3 * i + 2];
		cloud_in->points.push_back(p);
	}
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(200);
	//float plane_tolerance = 0.05f;
	seg.setDistanceThreshold(plane_tolerance);//(0.05);
	seg.setInputCloud(cloud_in);
	seg.segment(*inliers, *coefficients);
	/*Eigen::Vector3f normal;
	normal[0] = coefficients->values[0];
	normal[1] = coefficients->values[1];
	normal[2] = coefficients->values[2];
	normal.normalize();
	Eigen::Hyperplane<float, 3> plane(normal, coefficients->values[3]);
	int count = 0;
	for (size_t i = 0; i < cloud_in->points.size(); ++i)
	{
		Eigen::Vector3f v = cloud_in->points[i].getVector3fMap();
		if (plane.absDistance(v) < 0.1f)
		{
			labels[i] = 0;
			++count;
		}
			
	}
	std::cout << "plane count: " << count << std::endl;
	std::cout << "plane size: " << inliers->indices.size() << std::endl;*/
	for (size_t i = 0; i < inliers->indices.size(); ++i)
	{
		labels[inliers->indices[i]] = 0;
	}
}

void loadMesh(std::string& path)
{
	pcl::io::loadPLYFile(path, mesh);
}

void process_key_down(const pcl::visualization::KeyboardEvent& event, void* v)
{
	if ((event.getKeyCode() == 'a' || event.getKeyCode() == 'A') && event.keyUp() == true)
	{
		if (sample_len == 0.0f)
			return;
		if (points_picked->points.size() < 2)
			return;
		Eigen::Vector3f vec = points_picked->points[1].getVector3fMap() - points_picked->points[0].getVector3fMap();
		scale = sample_len / vec.norm();
		for (int i = 0; i < points_picked->points.size(); ++i)
		{
			points_picked->points[i].x *= scale;
			points_picked->points[i].y *= scale;
			points_picked->points[i].z *= scale;
		}
		scalingMesh(scale);
		camPose.translation() = scale*camPose.translation();
		setViewerPose(viewer, camPose);
		//scalingMesh();
		//setViewerFromMesh(viewer, mesh);
	}
	if ((event.getKeyCode() == 's' || event.getKeyCode() == 'S') && event.keyUp() == true)
	{
		int s = saveMesh();
		//if (s == 1)
			//decimatingMesh();
	}
	if ((event.getKeyCode() == 'q' || event.getKeyCode() == 'Q'))
	{
		loop = false;
	}
	if ((event.getKeyCode() == 'c' || event.getKeyCode() == 'C')&& event.keyUp() == true)
	{
		if (cur_view == img_paths.size())
			cur_view = 0;
		img_cur = cv::imread(img_paths[cur_view], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		img_origin_w = img_cur.cols;
		img_origin_h = img_cur.rows;
		float width = (float)img_cur.cols / 2 + 0.5f;
		float height = (float)img_cur.rows / 2 + 0.5f;
		cv::resize(img_cur, img_cur, cv::Size(width, height));
		cv::imshow("color image", img_cur);
		cv::waitKey(10);
		cv::setMouseCallback("color image", onMouse, 0);
		setViewerPose(viewer, camPoses[cur_view]);
		calib_mat = calib_mats[cur_view];
		inverse_calib_mat = inverse_calib_mats[cur_view];
		camPose = camPoses[cur_view];
		cur_view++;
	}
}

void process_mouse(const pcl::visualization::MouseEvent& event, void* v)
{
	if (event.getButton() == pcl::visualization::MouseEvent::RightButton)
	{
		if (event.getType() == pcl::visualization::MouseEvent::MouseButtonPress)
		{
			int px = event.getX();
			int py = event.getY();
			if (points_picked->size() > 1)
				points_picked->resize(0);
			pickingMesh(px, py, viewer, mesh, points_picked);
		}
	}
}

void initCamParams()
{
	camPoses.resize(0);
	world_to_cams.resize(0);
	calib_mats.resize(0);
	inverse_calib_mats.resize(0);
	img_paths.resize(0);
}

void createViewer(int x, int y, int width, int height, std::string &caption)
{
	viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer(caption));
	viewer->setWindowName(caption);
	viewer->setBackgroundColor(0, 0, 0.15);
	viewer->addCoordinateSystem(1.0, "world");
	viewer->initCameraParameters();
	viewer->setPosition(x, y);
	viewer->setSize(width, height);
	viewer->setCameraClipDistances(0.0, 10.0);
	viewer->registerKeyboardCallback(&process_key_down);
	viewer->registerMouseCallback(&process_mouse);
	setViewerPose(viewer, camPose);
}

void drawScene(int nType)
{
	while (loop)
	{ 
		viewer->removePolygonMesh("mesh");
		viewer->removeAllPointClouds();
		viewer->removeText3D("length_picked");
		viewer->addPolygonMesh(mesh, "mesh");
		if (points_picked->points.size() > 0)
		{
			viewer->addPointCloud<pcl::PointXYZRGB>(points_picked, "picked");
			viewer->setPointCloudRenderingProperties(0, 5.0f, "picked");
			if (points_picked->points.size() == 2)
			{
				Eigen::Vector3f vec = points_picked->points[1].getVector3fMap() - points_picked->points[0].getVector3fMap();
				float s = sample_len / vec.norm();
				char len_buf[512];
				//sprintf(len_buf, "sample_length:%f, length:%f, scale2:%f, view_count:%d\n",  sample_len, vec.norm(), s, n_views);
				sprintf(len_buf, "scale:%f, length:%f, picked count:%d\n", scale, vec.norm(), points_picked->points.size());
				std::string str_len = len_buf;
				viewer->addText(str_len, 10, 10, 16, 0.0, 1.0, 0.0, "length_picked");
			}
		}
		viewer->spinOnce(10);
	}
	viewer->close();
}

int getVertCount()
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud_in;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud_in);
	return cloud_in.points.size();
}

void getVertices(float* vertices, float* normals, int vert_count)
{
	pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_in;
	cloud_in.points.clear();
	pcl::fromPCLPointCloud2(mesh.cloud, cloud_in);
	size_t l = 0;
	for (int i = 0; i < cloud_in.points.size(); ++i)
	{
		Eigen::Vector3f vert = cloud_in.points[i].getVector3fMap();
		Eigen::Vector3f normal = cloud_in.points[i].getNormalVector3fMap();
		vertices[l] = vert[0]; normals[l] = normal[0]; ++l;
		vertices[l] = vert[1]; normals[l] = normal[1]; ++l;
		vertices[l] = vert[2]; normals[l] = normal[2]; ++l;
	}
}