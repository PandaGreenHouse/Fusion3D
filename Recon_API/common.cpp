#include "common.h"

On3DProgressCallback g_p3DProgressCallback = NULL;
void log_message(AppSettings const& app_conf, std::string const& message)
{
	std::cout << message << std::endl;
	if (app_conf.sceneSettings.log_file.empty())
		return;

	std::string fname = util::fs::join_path(app_conf.sceneSettings.path_scene, app_conf.sceneSettings.log_file);
	std::ofstream out(fname.c_str(), std::ios::app);
	if (!out.good())
		return;

	time_t rawtime;
	std::time(&rawtime);
	struct std::tm* timeinfo;
	timeinfo = std::localtime(&rawtime);
	char timestr[20];
	std::strftime(timestr, 20, "%Y-%m-%d %H:%M:%S", timeinfo);

	out << timestr << "  " << message << std::endl;
	out.close();
}

void clipping(mve::Scene::ViewList& views, mve::TriangleMesh::VertexList& verts)
{
	math::Vec3f vCenter(0.0f, 0.0f, 0.0f);
	std::vector<math::Vec3f> cam_positions;
	cam_positions.clear();
	for (int i = 0; i < views.size(); ++i)
	{
		float position[3];
		views.at(i).get()->get_camera().fill_camera_pos(position);
		math::Vec3f pos(position[0], position[1], position[2]);
		vCenter = ((float)i*vCenter + pos) / (i + 1);
		cam_positions.push_back(pos);
	}
	float radius = 0.0f;
	for (int i = 0; i < cam_positions.size(); ++i)
	{
		math::Vec3f vec = vCenter - cam_positions[i];
		if (vec.norm() > radius)
			radius = vec.norm();
	}
	mve::TriangleMesh::VertexList verts_clipped;
	verts_clipped.clear();
	for (size_t i = 0; i < verts.size(); ++i)
	{
		math::Vec3f vert = verts.at(i);
		math::Vec3f vec = vert - vCenter;
		if (vec.norm() < radius)
		{
			verts_clipped.push_back(vert);
		}
	}
	std::swap(verts_clipped, verts);
}

void extract_obj(mve::TriangleMesh::VertexList& verts, float plane_tolerance)
{
	std::vector<std::vector<float>> points_origin, points_result;
	points_origin.resize(verts.size());
	for (size_t i = 0; i < verts.size(); ++i)
	{
		points_origin[i].resize(3);
		points_origin[i][0] = verts[i][0];
		points_origin[i][1] = verts[i][1];
		points_origin[i][2] = verts[i][2];
	}
	points_result.clear();
	extractObj(points_origin, plane_tolerance, 0.02f);
	verts.resize(points_result.size());
	for (size_t i = 0; i < points_result.size(); ++i)
	{
		verts[i][0] = points_result[i][0];
		verts[i][1] = points_result[i][1];
		verts[i][2] = points_result[i][2];
	}
}

void extract_obj(std::string& scene_path, std::string& input_path, std::string& out_path, float plane_tolerance, float cluster_tolerance)
{
	/* Load scene. */
	mve::Scene::Ptr scene;
	try {
		scene = mve::Scene::create(scene_path);
	}
	catch (std::exception& e) {
		std::cerr << "Error loading scene: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	mve::Scene::ViewList const& views = scene->get_views();
	std::vector<float> cam_positions;
	cam_positions.clear();
	cam_positions.resize(3 * views.size());
	for (size_t i = 0; i < views.size(); ++i)
	{
		float position[3];
		views[i]->get_camera().fill_camera_pos(position);
		cam_positions[3 * i] = position[0];
		cam_positions[3 * i+1] = position[1];
		cam_positions[3 * i+2] = position[2];
		std::cout << "cam position:" << position[0] << "," << position[1] << "," << position[2] << std::endl;
	}
	extractObj(input_path, out_path, cam_positions, plane_tolerance, cluster_tolerance);
}

void extractObj(mve::Scene::Ptr scene, float plane_tolerance, mve::TriangleMesh::Ptr pset_in, mve::TriangleMesh::Ptr pset_out)
{
	//mve::TriangleMesh::Ptr pset(mve::TriangleMesh::create());
	mve::TriangleMesh::VertexList& verts(pset_in->get_vertices());
	mve::TriangleMesh::NormalList& vnorm(pset_in->get_vertex_normals());
	mve::TriangleMesh::ColorList& vcolor(pset_in->get_vertex_colors());
	mve::TriangleMesh::ValueList& vvalues(pset_in->get_vertex_values());
	mve::TriangleMesh::ConfidenceList& vconfs(pset_in->get_vertex_confidences());
	std::cout << "vert count: " << verts.size() << std::endl;
	
	mve::Scene::ViewList const& views = scene->get_views();
	std::cout << "view count: " << views.size() << std::endl;
	std::vector<math::Vec3f> cam_positions;
	cam_positions.resize(views.size());
	math::Vec3f vCenter(0.0f, 0.0f, 0.0f);
	for (size_t i = 0; i < views.size(); ++i)
	{
		float position[3];
		views[i]->get_camera().fill_camera_pos(position);
		cam_positions[i][0] = position[0];
		cam_positions[i][1] = position[1];
		cam_positions[i][2] = position[2];
		vCenter += cam_positions[i];
	}
	vCenter /= views.size();
	std::cout << "view count: " << views.size() << std::endl;
	float radius = 0.0f;
	for (size_t i = 0; i < cam_positions.size(); ++i)
	{
		math::Vec3f vec = cam_positions[i] - vCenter;
		if (vec.norm() > radius)
		{
			radius = vec.norm();
		}
	}
	std::vector<int> indices;
	indices.clear();
	for (size_t i = 0; i < verts.size(); ++i)
	{
		math::Vec3f vec = verts[i] - vCenter;
		if (vec.norm() > radius)
			continue;
		indices.push_back(i);
	}
	std::cout << "original vertex count: " << verts.size() << std::endl;
	std::cout << "clipped count: " << indices.size() << std::endl;
	mve::TriangleMesh::VertexList verts_clipped;
	mve::TriangleMesh::NormalList vnorm_clipped;
	mve::TriangleMesh::ColorList vcolor_clipped;
	mve::TriangleMesh::ValueList vvalues_clipped;
	mve::TriangleMesh::ConfidenceList vconfs_clipped;
	verts_clipped.clear();
	vnorm_clipped.clear();
	vcolor_clipped.clear();
	vvalues_clipped.clear();
	vconfs_clipped.clear();
	for (size_t i = 0; i < indices.size(); ++i)
	{
		verts_clipped.push_back(verts[indices[i]]);
		vnorm_clipped.push_back(vnorm[indices[i]]);
		vcolor_clipped.push_back(vcolor[indices[i]]);
		vvalues_clipped.push_back(vvalues[indices[i]]);
		vconfs_clipped.push_back(vconfs[indices[i]]);
	}
	std::vector<float> vert_data;
	std::vector<int> labels;
	vert_data.resize(3 * verts_clipped.size());
	labels.resize(verts_clipped.size(),1);
	for (size_t i = 0; i < verts_clipped.size(); ++i)
	{
		vert_data[3 * i] = verts_clipped[i][0];
		vert_data[3 * i+1] = verts_clipped[i][1];
		vert_data[3 * i+2] = verts_clipped[i][2];
	}
	segmentPlane(vert_data, plane_tolerance, labels);
	mve::TriangleMesh::VertexList& verts_out(pset_out->get_vertices());
	mve::TriangleMesh::NormalList& vnorm_out(pset_out->get_vertex_normals());
	mve::TriangleMesh::ColorList& vcolor_out(pset_out->get_vertex_colors());
	mve::TriangleMesh::ValueList& vvalues_out(pset_out->get_vertex_values());
	mve::TriangleMesh::ConfidenceList& vconfs_out(pset_out->get_vertex_confidences());
	size_t count = 0;
	for (size_t i = 0; i < labels.size(); ++i)
	{
		if (labels[i] == 0)
		{
			++count;
			continue;
		}
		verts_out.push_back(verts_clipped[i]);
		vnorm_out.push_back(vnorm_clipped[i]);
		vcolor_out.push_back(vcolor_clipped[i]);
		vvalues_out.push_back(vvalues_clipped[i]);
	}
}