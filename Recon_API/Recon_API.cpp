// Recon_API.cpp : Defines the exported functions for the DLL application.
//
#include "common.h"
//#include "stdafx.h"
#include "Recon_API.h"
//#include "cv_matcher.h"
//using namespace std;
//using namespace cv;
AppSettings appSettings;
#define MAX_PIXELS1 3000000
#define MAX_PIXELS2 6000000

void init_recon(std::string& img_set_path, std::string& scene_path, int match_mode, int mvsType,
				int max_pixels_level, int dm_input_scale, int dm_out_scale, bool use_shading, 
				float plane_tolerance, float cluster_tolerance, On3DProgressCallback pProgressCallback)
{
	switch (max_pixels_level)
	{
	case 1:
		appSettings.sceneSettings.max_pixels = MAX_PIXELS1;
		break;
	case 2:
		appSettings.sceneSettings.max_pixels = MAX_PIXELS2;
		break;
	}
	appSettings.sceneSettings.path_image = img_set_path;
	appSettings.sceneSettings.path_scene = scene_path;
	appSettings.sfmSettings.match_mode = match_mode;
	appSettings.psetSettings.pset_name1 = scene_path + "/obj.ply";
	appSettings.smvsSettings.input_scale = dm_input_scale;
	appSettings.smvsSettings.output_scale = dm_out_scale;
	appSettings.smvsSettings.use_shading = use_shading;
	appSettings.plane_tolerance = plane_tolerance;
	appSettings.cluster_tolerance = cluster_tolerance;
	g_p3DProgressCallback = pProgressCallback;
}

void make_scene()
{
	make_scene(appSettings);
}

void sfm_recon()
{
	sfm_reconstruct(appSettings);
}

void dm_recon(int mvsType)
{
	if (mvsType == 0)
	{
		dmrecon(appSettings);
		scene2pset(appSettings);
	}
	if (mvsType == 1)
	{
		smvsrecon(appSettings);
	}
}

void fss_recon()
{
	fssrecon(appSettings);
	mesh_clean(appSettings);
}

void tex_recon(std::string& scene_path, float s)
{
	std::string in_scene = scene_path + "::original";
	std::string in_mesh = scene_path + "/clean.ply";
	std::string out_prefix = scene_path + "/prefix";
	texRecon(in_scene, in_mesh, out_prefix, s);
}

void recon_all(std::string& img_set_path, std::string& scene_path, int match_mode, int mvsType)
{
	appSettings.sceneSettings.path_image = img_set_path;
	appSettings.sceneSettings.path_scene = scene_path;
	appSettings.sfmSettings.match_mode = match_mode;
	appSettings.psetSettings.pset_name1 = scene_path + "/obj.ply";
	//appSettings.smvsSettings.input_scale = 2;
	//appSettings.smvsSettings.output_scale = 2;
	make_scene(appSettings);
	
	g_p3DProgressCallback(7, 100, "make_scene is done");
	
	g_p3DProgressCallback(7, 0, "sfm starts");
	sfm_reconstruct(appSettings);
	g_p3DProgressCallback(37, 100, "sfm done!");
	g_p3DProgressCallback(37, 0, "dm_recon starts");
	if (mvsType == 0)
	{
		dmrecon(appSettings);
		scene2pset(appSettings);
	}
	if (mvsType == 1)
	{
		smvsrecon(appSettings);
	}
	g_p3DProgressCallback(70, 100, "dm_recon done!");
	g_p3DProgressCallback(70, 0, "meshing starts.");
	fssrecon(appSettings);
	mesh_clean(appSettings);
	g_p3DProgressCallback(90, 100, "meshing done!");
	g_p3DProgressCallback(90, 0, "texturing starts.");
	std::string in_scene = scene_path + "::original";//"f:/dataset/foot/scene_mve::original";
	std::string out_prefix = scene_path + "/prefix";//"f:/dataset/foot/scene_mve/prefix";
	std::string in_mesh = scene_path + "/clean.ply";//"f:/dataset/foot/scene_mve/clean.ply";
	texRecon(in_scene, in_mesh, out_prefix, 0.0f);
	g_p3DProgressCallback(100, 100, "texturing done!");
}

int loadScene(std::string &path)
{
	std::string scene_path = path;
	if (path == "")
		return 0;
	/*std::string in_mesh = scene_path + "/clean.ply";
	if (util::fs::file_exists(in_mesh.c_str()) == false)
		return 1;
	loadMesh(in_mesh);*/
	/* Load scene. */
	mve::Scene::Ptr scene;
	try {
		scene = mve::Scene::create(scene_path);
	}
	catch (std::exception& e) {
		std::cerr << "Error loading scene: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
		return 1;
	}
	mve::Scene::ViewList const& views = scene->get_views();
	std::vector<float> cam_positions;
	std::vector<std::string> img_paths;
	img_paths.resize(0);
	cam_positions.clear();
	cam_positions.resize(3 * views.size());
	std::vector<float> focal_lens;
	focal_lens.resize(views.size());
	initCamParams();
	int img_width = appSettings.sceneSettings.width;
	int img_height = appSettings.sceneSettings.height;
	float avg_focal_len = 0.0f;
	for (size_t i = 0; i < views.size(); ++i)
	{
		std::string img_path = views[i]->get_directory() + "/" + views[i]->get_images().at(0).filename;//views[i]->get_directory() + "/" + appSettings.sceneSettings.img_extension;
		auto img = cv::imread(img_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		img_width = img.cols;
		img_height = img.rows;
		addImgPath(img_path);
		float position[3], f_len;
		views[i]->get_camera().fill_camera_pos(position);
		cam_positions[3 * i] = position[0];
		cam_positions[3 * i + 1] = position[1];
		cam_positions[3 * i + 2] = position[2];
		focal_lens[i] = views[i]->get_camera().flen;
		float rot[9], worldToCam[16], calib[9], inverse_calib[9];
		views[i]->get_camera().fill_cam_to_world_rot(rot);
		views[i]->get_camera().fill_world_to_cam(worldToCam);
		views[i]->get_camera().fill_calibration(calib, img_width, img_height);
		views[i]->get_camera().fill_inverse_calibration(inverse_calib, img_width, img_height);
		f_len = views[i]->get_camera().flen;
		getCamParams(position, rot, worldToCam, calib, inverse_calib, views[i]->get_camera().flen);
		avg_focal_len = (i*avg_focal_len + f_len) / (i + 1);
	}
	appSettings.est_focal_length = avg_focal_len;
	return 3;
}

void scalingModel(float real_focal_len)
{
	appSettings.device_focal_length = real_focal_len;
	float s = real_focal_len / appSettings.est_focal_length;
	scalingMesh(s);
}

void scalingTexturedMesh()
{
	std::string tex_mesh_path = appSettings.sceneSettings.path_scene + "/prefix.obj";
	
}