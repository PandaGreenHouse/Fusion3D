#include "util.h"
#include "Simplify.h"
/*
created at Jan 13, 2022
*/
bool intersectTri(Eigen::Vector3f &v0, Eigen::Vector3f &v1, Eigen::Vector3f &v2,
	Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, float &u, float &v, float &dist, Eigen::Vector3f &interPos)
{
	Eigen::Vector3f e1 = v1 - v0;
	Eigen::Vector3f e2 = v2 - v0;
	Eigen::Vector3f q = vRayDir.cross(e2);
	float a = e1.dot(q);//D3DXVec3Dot(&e1,&q);//e1.dot(q);
	Eigen::Vector3f s = vRayPos - v0;
	Eigen::Vector3f r = s.cross(e1);
	//D3DXVec3Cross(&r,&s,&e1);//s.cross(e1);
	// Barycentric vertex weights
	u = s.dot(q) / a;
	v = vRayDir.dot(r) / a;
	float w = 1.0f - (u + v);//weight[0] = 1.0f - (weight[1] + weight[2]);
	dist = e2.dot(r) / a;
	static const float epsilon = 1e-7f;
	static const float epsilon2 = 1e-10f;
	if ((a <= epsilon) || (u < -epsilon2) ||
		(v < -epsilon2) || (w < -epsilon2) ||
		(dist <= 0.0f)) {
		// The ray is nearly parallel to the triangle, or the
		// intersection lies outside the triangle or behind
		// the ray origin: "infinite" distance until intersection.
		return false;
	}
	else {
		interPos = v0 + u*e1 + v*e2;
		return true;
	}
}

bool getIntersection(Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, pcl::PolygonMesh &mesh, Eigen::Vector3f &intersection)
{
	bool bIntersect = false;
	pcl::PointCloud<pcl::PointXYZRGB> cloud_in;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud_in);
	float u, v;
	Eigen::Vector3f pts[3];
	float minDist = std::numeric_limits<float>::max();
	int pickId = -1;
	for (size_t i = 0; i < mesh.polygons.size(); ++i)
	{
		//int indices[3];
		Eigen::Vector3f points[3];
		for (int j = 0; j < 3; ++j)
		{
			int id = mesh.polygons[i].vertices[j];
			points[j] = cloud_in.points[id].getVector3fMap();
		}
		float u1, v1, dist;
		Eigen::Vector3f interPos;
		if (intersectTri(points[0], points[1], points[2], vRayPos, vRayDir, u1, v1, dist, interPos))
		{
			if (minDist > dist)
			{
				for (int k = 0; k < 3; ++k)
				{
					pts[k] = points[k];
				}
				u = u1;
				v = v1;
				pickId = i;
				minDist = dist;
				intersection = interPos;
				bIntersect = true;
			}
		}
	}
	return bIntersect;
}

bool pickingMesh(int px, int py, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PolygonMesh &mesh,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_picked)
{
	std::vector<pcl::visualization::Camera> cams;
	viewer->getCameras(cams);
	int width = cams[0].window_size[0];
	int height = cams[0].window_size[1];
	float cx = (float)width / 2;
	float cy = (float)height / 2;
	float y_max = tan(cams[0].fovy / 2);
	float x_max = y_max*(float)width / height;
	float x = (float)px - cx;
	float y = (float)py - cy;
	Eigen::Vector3f p;
	float w = 0.5f*(float)width;
	float h = 0.5f*(float)height;
	p[0] = -x_max*(float)x / w;
	p[1] = y_max*(float)y / h;
	p[2] = 1.0f;
	Eigen::Affine3f pose = viewer->getViewerPose();
	Eigen::Vector3f vRayDir = pose.linear()*p;
	Eigen::Vector3f vRayPos = pose.translation();
	vRayDir.normalize();
	Eigen::Vector3f intersection;
	if (getIntersection(vRayPos, vRayDir, mesh, intersection))
	{
		//if(points_picked->size()==2)
		//points_picked->points.resize(0);
		pcl::PointXYZRGB p;
		p.x = intersection[0];
		p.y = intersection[1];
		p.z = intersection[2];
		p.r = 255;
		p.g = 0;
		p.b = 0;
		points_picked->points.push_back(p);
		return true;
	}
	return false;
}

bool pickingMesh(Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, pcl::PolygonMesh &mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_picked)
{
	Eigen::Vector3f intersection;
	if (getIntersection(vRayPos, vRayDir, mesh, intersection))
	{
		//if(points_picked->size()==2)
		//points_picked->points.resize(0);
		pcl::PointXYZRGB p;
		p.x = intersection[0];
		p.y = intersection[1];
		p.z = intersection[2];
		p.r = 255;
		p.g = 0;
		p.b = 0;
		points_picked->points.push_back(p);
		return true;
	}
	return false;
}

void drawPickingsOnImg(cv::Mat& img, int px, int py)
{
	if (px < 1 || px >(img.cols - 2))
		return;
	if (py < 1 || py >(img.rows - 2))
		return;
	for (int i = 0; i < 2; ++i)
	{
		for (int iy = -1; iy < 2; ++iy)
		{
			for (int ix = -1; ix < 2; ++ix)
			{
				img.at<cv::Vec3b>(py + iy, px + ix) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::imshow("color image", img);
	cv::waitKey(10);
}
bool pickingMesh(Eigen::Vector3f &vRayPos, Eigen::Vector3f &vRayDir, pcl::PolygonMesh &mesh, pcl::PointXYZRGB& point_picked)
{
	Eigen::Vector3f intersection;
	if (getIntersection(vRayPos, vRayDir, mesh, intersection))
	{
		point_picked.x = intersection[0];
		point_picked.y = intersection[1];
		point_picked.z = intersection[2];
		point_picked.r = 255;
		point_picked.g = 0;
		point_picked.b = 0;
		return true;
	}
	return false;
}

void setViewerPose(pcl::visualization::PCLVisualizer::Ptr viewer, Eigen::Affine3f &pose)
{
	Eigen::Vector3f pos_vector = pose.translation();
	Eigen::Vector3f vLookAt = pos_vector + pose.linear()*Eigen::Vector3f::UnitZ();
	Eigen::Vector3f up_vec = pose.linear()*(-Eigen::Vector3f::UnitY());
	viewer->setCameraPosition(pos_vector[0], pos_vector[1], pos_vector[2], vLookAt[0], vLookAt[1], vLookAt[2], up_vec[0], up_vec[1], up_vec[2]);
}

void setViewerFromMesh(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PolygonMesh &mesh)
{
	pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud_xyz);
	float maxDist = 0.0f;
	Eigen::Vector3f vCenter(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < cloud_xyz.points.size(); ++i)
	{
		vCenter = (i*vCenter + cloud_xyz.points[i].getVector3fMap()) / (i + 1);
	}
	for (int i = 0; i < cloud_xyz.points.size(); ++i)
	{
		Eigen::Vector3f vec = cloud_xyz.points[i].getVector3fMap() - vCenter;
		if (maxDist < vec.norm())
			maxDist = vec.norm();
	}
	Eigen::Vector3f pos_vector = vCenter + 2 * maxDist*Eigen::Vector3f::UnitZ();
	Eigen::Vector3f vLookAt = vCenter;
	Eigen::Vector3f up_vec = Eigen::Vector3f::UnitY();
	viewer->setCameraPosition(pos_vector[0], pos_vector[1], pos_vector[2], vLookAt[0], vLookAt[1], vLookAt[2], up_vec[0], up_vec[1], up_vec[2]);
}

/*
created at 01/22/2019
*/
void decimateMesh(pcl::PolygonMesh& mesh_in, pcl::PolygonMesh& mesh_out)
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud_in;
	pcl::fromPCLPointCloud2(mesh_in.cloud, cloud_in);
	std::vector<Eigen::Vector3f> vertices_in;
	std::vector<int> indices_in;
	vertices_in.resize(0);
	indices_in.resize(0);
	for (int i = 0; i < cloud_in.points.size(); ++i)
	{
		Eigen::Vector3f v = cloud_in.points.at(i).getVector3fMap();
		vertices_in.push_back(v);
	}
	for (int i = 0; i < mesh_in.polygons.size(); ++i)
	{
		pcl::Vertices vertices = mesh_in.polygons[i];
		int id = vertices.vertices[0];
		indices_in.push_back(id);
		id = vertices.vertices[1];
		indices_in.push_back(id);
		id = vertices.vertices[2];
		indices_in.push_back(id);
	}
	Simplify::importMesh(vertices_in, indices_in);
	int target_count = mesh_in.polygons.size() / 200;//10,5;
	double agressiveness = 11;// 7;
	Simplify::simplify_mesh(target_count, agressiveness, true);
	std::vector<Eigen::Vector3f> vertices_decimated;
	std::vector<int> indices_decimated;
	Simplify::exportMesh(vertices_decimated, indices_decimated);
	pcl::PointCloud<pcl::PointXYZ> cloud_decimated;
	for (int i = 0; i < vertices_decimated.size(); ++i)
	{
		pcl::PointXYZ p;
		p.x = vertices_decimated[i][0];
		p.y = vertices_decimated[i][1];
		p.z = vertices_decimated[i][2];
		cloud_decimated.push_back(p);
	}
	pcl::toPCLPointCloud2(cloud_decimated, mesh_out.cloud);
	for (int i = 0; i < indices_decimated.size() / 3; ++i)
	{
		pcl::Vertices vertices;
		vertices.vertices.resize(3);
		vertices.vertices[0] = indices_decimated[3 * i];
		vertices.vertices[1] = indices_decimated[3 * i + 1];
		vertices.vertices[2] = indices_decimated[3 * i + 2];
		mesh_out.polygons.push_back(vertices);
	}
}