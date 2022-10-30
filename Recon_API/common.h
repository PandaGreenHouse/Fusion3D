#pragma once
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <stdexcept>
#include <cerrno>
#include <set>

#include "util/system.h"
#include "util/timer.h"
#include "util/string.h"
#include "util/arguments.h"
#include "util/file_system.h"
#include "util/tokenizer.h"

#include "math/algo.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "math/octree_tools.h"

#include "mve/bundle.h"
#include "mve/bundle_io.h"
#include "mve/image.h"
#include "mve/image_tools.h"
#include "mve/image_io.h"
#include "mve/image_exif.h"
#include "mve/depthmap.h"
#include "mve/mesh.h"
#include "mve/mesh_info.h"
#include "mve/mesh_io.h"
#include "mve/mesh_io_ply.h"
#include "mve/mesh_tools.h"
#include "mve/scene.h"
#include "mve/view.h"

#include "sfm/nearest_neighbor.h"
#include "sfm/feature_set.h"
#include "sfm/bundler_common.h"
#include "sfm/bundler_features.h"
#include "sfm/bundler_matching.h"
#include "sfm/bundler_tracks.h"
#include "sfm/bundler_init_pair.h"
#include "sfm/bundler_intrinsics.h"
#include "sfm/bundler_incremental.h"

#include "dmrecon/settings.h"
#include "dmrecon/dmrecon.h"

#include "smvsrecon/thread_pool.h"
#include "smvsrecon/stereo_view.h"
#include "smvsrecon/depth_optimizer.h"
#include "smvsrecon/mesh_generator.h"
#include "smvsrecon/view_selection.h"
#include "smvsrecon/sgm_stereo.h"

#include "fssr/mesh_clean.h"
#include "fssr/sample_io.h"
#include "fssr/iso_octree.h"
#include "fssr/iso_surface.h"
#include "fssr/hermite.h"
#include "fssr/defines.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "fancy_progress_printer.h"
#include "extractor_api.h"
/*#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"
#include "arguments.h"*/

typedef void(*On3DProgressCallback)(int total_progress, int sub_progress, char* msg);

extern On3DProgressCallback g_p3DProgressCallback;

#define THUMBNAIL_SIZE				50
#define RAND_SEED_MATCHING			0
#define RAND_SEED_SFM				0

#define BUNDLE_PATH					"bundle/"
#define PS_BUNDLE_LOG				"coll.log"
#define PS_IMAGE_DIR				"images/"
#define PS_UNDIST_DIR				"undistorted/"
#define BUNDLER_FILE_LIST			"list.txt"
#define BUNDLER_IMAGE_DIR			""
#define VIEWS_DIR					"views/"

typedef std::vector<std::string> StringVector;
typedef std::vector<util::fs::File> FileVector;
typedef FileVector::iterator FileIterator;


enum ProgressStyle
{
	PROGRESS_SILENT,
	PROGRESS_SIMPLE,
	PROGRESS_FANCY
};

using namespace std;
using namespace cv;
//FancyProgressPrinter fancyProgressPrinter;

struct SceneSettings
{
	std::string path_image;
	std::string path_scene;
	std::string log_file = "log.txt";
	int bundle_id = 0;
	bool import_orig = false;
	bool skip_invalid = true;
	bool images_only = false;
	bool append_images = false;
	int max_pixels = std::numeric_limits<int>::max();
	std::string img_extension = "png";//added at 2022.07.24. it is defined png or jpg
									  /* Computed values. */
	std::string path_bundle;
	std::string path_views;
	int width;
	int height;
};

struct SfmSettings
{
	std::string original_name = "original";
	std::string undistorted_name = "undistorted";
	std::string exif_name = "exif";
	std::string prebundle_file = "prebundle.sfm";
	std::string survey_file;
	//std::string log_file = "log.txt";
	int max_image_size = 6000000;
	bool lowres_matching = true;
	bool normalize_scene = false;
	bool skip_sfm = false;
	bool always_full_ba = false;
	bool fixed_intrinsics = false;
	//bool shared_intrinsics = false;
	bool intrinsics_from_views = false;
	int video_matching = 0;
	float track_error_thres_factor = 10.0f;
	float new_track_error_thres = 0.01f;
	int initial_pair_1 = -1;
	int initial_pair_2 = -1;
	int min_views_per_track = 3;
	bool cascade_hashing = false;
	bool verbose_ba = false;
	int match_mode = 0;//added at 2022//08/02
};

struct DmSettings
{
	std::string ply_dest = "recon";
	int master_id = -1;
	std::vector<int> view_ids;
	bool force_recon = false;
	bool write_ply = false;
#ifdef _WIN32
	ProgressStyle progress_style = PROGRESS_SIMPLE;
#else
	ProgressStyle progress_style = PROGRESS_FANCY;
#endif  // _WIN32
	mvs::Settings mvs;
};

struct Scene2PsetSettings
{
	std::string pset_name, pset_name1;
	std::string dmname = "depth-L0";
	std::string image = "undistorted";
	std::string mask;
	std::string aabb;
	bool with_normals = false;
	bool with_scale = false;
	bool with_conf = false;
	bool poisson_normals = false;
	float min_valid_fraction = 0.0f;
	float scale_factor = 2.5f; /* "Radius" of MVS patch (usually 5x5). */
	std::vector<int> ids;
};

struct SmvsreconSettings
{
	std::string image_embedding = "undistorted";
	float regularization = 1.0;
	int output_scale = 3;//2
	int input_scale = 3;//2
	int debug_lvl = 0;
	std::size_t num_neighbors = 6;
	std::size_t min_neighbors = 3;
	// 	std::size_t max_pixels = 1700000;
	std::size_t num_threads = std::thread::hardware_concurrency();
	bool use_shading = false;
	float light_surf_regularization = 0.0f;
	bool gamma_correction = false;
	bool recon_only = false;
	bool cut_surface = true;
	bool create_triangle_mesh = false;
	std::string aabb_string = "";
	bool simplify = false;
	bool use_sgm = true;
	float sgm_min = 0.0f;
	float sgm_max = 0.0f;
	int sgm_scale = 1;
	std::string sgm_range = "";
	// 	bool force_recon = false;
	bool force_sgm = false;
	bool full_optimization = false;
	bool clean_scene = false;
	math::Vec3f aabb_min = math::Vec3f(0.0f);
	math::Vec3f aabb_max = math::Vec3f(0.0f);
	std::string out_pset_name;
};

struct FssreconSettings
{
	std::vector<std::string> in_files;
	std::string out_mesh;
	int refine_octree = 0;
	fssr::InterpolationType interp_type = fssr::INTERPOLATION_CUBIC;
};

struct MeshCleanSettings
{
	std::string out_mesh_clean;
	bool clean_degenerated = true;
	bool delete_scale = false;
	bool delete_conf = false;
	bool delete_colors = false;
	float conf_threshold = 1.0f;
	float conf_percentile = -1.0f;
	int component_size = 1000;
};

struct AppSettings
{
	SceneSettings sceneSettings;
	SfmSettings   sfmSettings;
	DmSettings    dmSettings;
	Scene2PsetSettings psetSettings;
	SmvsreconSettings smvsSettings;
	FssreconSettings FssreconSettings;
	MeshCleanSettings meshCleanSettings;
	float plane_tolerance = 0.05f;
	float cluster_tolerance = 0.02f;
	float device_focal_length = 3.5f;
	float est_focal_length = 1.0f;
};


void log_message(AppSettings const& conf, std::string const& message);
int make_scene(AppSettings& appSettings);
void sfm_reconstruct(AppSettings const& conf);
int dmrecon(AppSettings& conf);
int scene2pset(AppSettings& conf);
int fssrecon(AppSettings& conf/*, fssr::SampleIO::Options const& pset_opts*/);
int mesh_clean(AppSettings& conf);
int smvsrecon(AppSettings& conf);
void texRecon(std::string &in_scene, std::string &in_mesh, std::string &out_prefix, float s);
void clipping(mve::Scene::ViewList& views, mve::TriangleMesh::VertexList& verts);
void extract_obj(std::string& scene_path, std::string& input_path, std::string& out_path, float plane_tolerance, float cluster_tolerance);
void extractObj(mve::Scene::Ptr scene, float plane_tolerance, mve::TriangleMesh::Ptr pset_in, mve::TriangleMesh::Ptr pset_out);