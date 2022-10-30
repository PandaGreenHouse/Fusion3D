/*
* Copyright (C) 2015, Simon Fuhrmann
* TU Darmstadt - Graphics, Capture and Massively Parallel Computing
* All rights reserved.
*
* This software may be modified and distributed under the terms
* of the BSD 3-Clause license. See the LICENSE.txt file for details.
*/

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

#include "cv_matcher.h"

#include "fancy_progress_printer.h"
#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"
#include "arguments.h"

//#include "util.h"
using namespace std;
using namespace cv;

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

FancyProgressPrinter fancyProgressPrinter;

struct AppSettings
{
	//make scene
	std::string path_image;
	std::string path_scene;
	int bundle_id = 0;
	bool import_orig = false;
	bool skip_invalid = true;
	bool images_only = false;
	bool append_images = false;
	int max_pixels = std::numeric_limits<int>::max();

	/* Computed values. */
	std::string path_bundle;
	std::string path_views;

	//sfmrecon
// 	std::string scene_path;
	std::string original_name = "original";
	std::string undistorted_name = "undistorted";
	std::string exif_name = "exif";
	std::string prebundle_file = "prebundle.sfm";
	std::string survey_file;
	std::string log_file = "log.txt";
	int max_image_size = 1000000;//6000000;
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
	
	//dmrecon
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
	//scene2pset
// 	std::string scenedir;
	std::string pset_name;
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

	//smvsrecon
// 	std::string scene_dname;
// 	std::vector<int> view_ids;
	std::string image_embedding = "undistorted";
	float regularization = 1.0;
	int output_scale = 2;
	int input_scale = 2;
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

	//fssrecon
	std::vector<std::string> in_files;
	std::string out_mesh;
	int refine_octree = 0;
	fssr::InterpolationType interp_type = fssr::INTERPOLATION_CUBIC;


	//meshclean
//	std::string in_mesh;
	std::string out_mesh_clean;
	bool clean_degenerated = true;
	bool delete_scale = false;
	bool delete_conf = false;
	bool delete_colors = false;
	float conf_threshold = 1.0f;
	float conf_percentile = -1.0f;
	int component_size = 1000;
};

std::string img_extension = "png";//added at 2022.07.24. it is defined png or jpg
template <typename T> T percentile(std::vector<T> const& in, float percent)
{
	/* copy the input vector because 'nth_element' will rearrange it */
	std::vector<T> copy = in;
	std::size_t n = static_cast<std::size_t>(percent / 100.0f * in.size());
	std::nth_element(copy.begin(), copy.begin() + n, copy.end());
	return copy[n];
}

void remove_low_conf_vertices(mve::TriangleMesh::Ptr mesh, float const thres)
{
	mve::TriangleMesh::ConfidenceList const& confs = mesh->get_vertex_confidences();
	std::vector<bool> delete_list(confs.size(), false);
	for (std::size_t i = 0; i < confs.size(); ++i)
	{
		if (confs[i] > thres)
			continue;
		delete_list[i] = true;
	}
	mesh->delete_vertices_fix_faces(delete_list);
}

void
poisson_scale_normals(mve::TriangleMesh::ConfidenceList const& confs,
	mve::TriangleMesh::NormalList* normals)
{
	if (confs.size() != normals->size())
		throw std::invalid_argument("Invalid confidences or normals");
	for (std::size_t i = 0; i < confs.size(); ++i)
		normals->at(i) *= confs[i];
}

void
aabb_from_string(std::string const& str,
	math::Vec3f* aabb_min, math::Vec3f* aabb_max);

// void
// aabb_from_string (std::string const& str,
//     math::Vec3f* aabb_min, math::Vec3f* aabb_max)
// {
//     util::Tokenizer tok;
//     tok.split(str, ',');
//     if (tok.size() != 6)
//     {
//         std::cerr << "Error: Invalid AABB given" << std::endl;
//         std::exit(EXIT_FAILURE);
//     }
// 
//     for (int i = 0; i < 3; ++i)
//     {
//         (*aabb_min)[i] = tok.get_as<float>(i);
//         (*aabb_max)[i] = tok.get_as<float>(i + 3);
//     }
//     std::cout << "Using AABB: (" << (*aabb_min) << ") / ("
//         << (*aabb_max) << ")" << std::endl;
// }

void
reconstruct(mve::Scene::Ptr scene, mvs::Settings settings)
{
	/*
	* Note: destructor of ProgressHandle sets status to failed
	* if setDone() is not called (this happens when an exception
	* is thrown in mvs::DMRecon)
	*/
	ProgressHandle handle(fancyProgressPrinter, settings);
	mvs::DMRecon recon(scene, settings);
	handle.setRecon(recon);
	recon.start();
	handle.setDone();
}

void
aabb_from_string(std::string const& str,
	math::Vec3f* aabb_min, math::Vec3f* aabb_max)
{
	util::Tokenizer tok;
	tok.split(str, ',');
	if (tok.size() != 6)
	{
		std::cerr << "Error: Invalid AABB given" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 3; ++i)
	{
		(*aabb_min)[i] = tok.get_as<float>(i);
		(*aabb_max)[i] = tok.get_as<float>(i + 3);
	}
	std::cout << "Using AABB: (" << (*aabb_min) << ") / ("
		<< (*aabb_max) << ")" << std::endl;
}

int
get_scale_from_max_pixels(mve::Scene::Ptr scene,
	AppSettings const& app_settings, mvs::Settings const& mvs_settings)
{
	mve::View::Ptr view = scene->get_view_by_id(mvs_settings.refViewNr);
	if (view == nullptr)
		return 0;

	mve::View::ImageProxy const* proxy = view->get_image_proxy(mvs_settings.imageEmbedding);
	if (proxy == nullptr)
		return 0;

	int const width = proxy->width;
	int const height = proxy->height;
	if (width * height <= app_settings.max_pixels)
		return 0;

	float const ratio = width * height / static_cast<float>(app_settings.max_pixels);
	float const scale = std::ceil(std::log(ratio) / std::log(4.0f));

	std::cout << "Setting scale " << scale << " for " << width << "x" << height << " image." << std::endl;

	return std::max(0, static_cast<int>(scale));
}

void
wait_for_user_confirmation(void)
{
	std::cerr << "-> Press ENTER to continue, or CTRL-C to exit." << std::endl;
	std::string line;
	std::getline(std::cin, line);
}

/* ---------------------------------------------------------------- */

void
read_noah_imagelist(std::string const& filename, StringVector& files)
{
	/*
	* The list of the original images is read from the list.txt file.
	*/
	std::ifstream in(filename.c_str(), std::ios::binary);
	if (!in.good())
	{
		std::cerr << "Error: Cannot read bundler list file!" << std::endl;
		std::cerr << "File: " << filename << std::endl;
		std::exit(EXIT_FAILURE);
	}

	while (true)
	{
		std::string file, dummy;
		in >> file;
		std::getline(in, dummy);
		if (file.empty())
			break;
		files.push_back(file);
	}

	in.close();
}

/* ---------------------------------------------------------------- */

mve::ByteImage::Ptr
load_8bit_image(std::string const& fname, std::string* exif)
{
	std::string lcfname(util::string::lowercase(fname));
	std::string ext4 = util::string::right(lcfname, 4);
	std::string ext5 = util::string::right(lcfname, 5);
	try
	{
		if (ext4 == ".jpg" || ext5 == ".jpeg")
			return mve::image::load_jpg_file(fname, exif);
		else if (ext4 == ".png" || ext4 == ".ppm"
			|| ext4 == ".tif" || ext5 == ".tiff")
			return mve::image::load_file(fname);
	}
	catch (...)
	{
	}

	return mve::ByteImage::Ptr();
}

/* ---------------------------------------------------------------- */

mve::RawImage::Ptr
load_16bit_image(std::string const& fname)
{
	std::string lcfname(util::string::lowercase(fname));
	std::string ext4 = util::string::right(lcfname, 4);
	std::string ext5 = util::string::right(lcfname, 5);
	try
	{
		if (ext4 == ".tif" || ext5 == ".tiff")
			return mve::image::load_tiff_16_file(fname);
		else if (ext4 == ".ppm")
			return mve::image::load_ppm_16_file(fname);
	}
	catch (...)
	{
	}

	return mve::RawImage::Ptr();
}

/* ---------------------------------------------------------------- */

mve::FloatImage::Ptr
load_float_image(std::string const& fname)
{
	std::string lcfname(util::string::lowercase(fname));
	std::string ext4 = util::string::right(lcfname, 4);
	try
	{
		if (ext4 == ".pfm")
			return mve::image::load_pfm_file(fname);
	}
	catch (...)
	{
	}

	return mve::FloatImage::Ptr();
}

/* ---------------------------------------------------------------- */

mve::ImageBase::Ptr
load_any_image(std::string const& fname, std::string* exif)
{
	mve::ByteImage::Ptr img_8 = load_8bit_image(fname, exif);
	if (img_8 != nullptr)
		return img_8;

	mve::RawImage::Ptr img_16 = load_16bit_image(fname);
	if (img_16 != nullptr)
		return img_16;

	mve::FloatImage::Ptr img_float = load_float_image(fname);
	if (img_float != nullptr)
		return img_float;

#pragma omp critical
	std::cerr << "Skipping file " << util::fs::basename(fname)
		<< ", cannot load image." << std::endl;
	return mve::ImageBase::Ptr();
}

/* ---------------------------------------------------------------- */

template <typename T>
void
find_min_max_percentile(typename mve::Image<T>::ConstPtr image,
	T* vmin, T* vmax)
{
	typename mve::Image<T>::Ptr copy = mve::Image<T>::create(*image);
	std::sort(copy->begin(), copy->end());
	*vmin = copy->at(copy->get_value_amount() / 10);
	*vmax = copy->at(9 * copy->get_value_amount() / 10);
}

/* ---------------------------------------------------------------- */

void
add_exif_to_view(mve::View::Ptr view, std::string const& exif)
{
	if (exif.empty())
		return;

	mve::ByteImage::Ptr exif_image = mve::ByteImage::create(exif.size(), 1, 1);
	std::copy(exif.begin(), exif.end(), exif_image->begin());
	view->set_blob(exif_image, "exif");
}

/* ---------------------------------------------------------------- */

mve::ByteImage::Ptr
create_thumbnail(mve::ImageBase::ConstPtr img)
{
	mve::ByteImage::Ptr image;
	switch (img->get_type())
	{
	case mve::IMAGE_TYPE_UINT8:
		image = mve::image::create_thumbnail<uint8_t>
			(std::dynamic_pointer_cast<mve::ByteImage const>(img),
				THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		break;

	case mve::IMAGE_TYPE_UINT16:
	{
		mve::RawImage::Ptr temp = mve::image::create_thumbnail<uint16_t>
			(std::dynamic_pointer_cast<mve::RawImage const>(img),
				THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		uint16_t vmin, vmax;
		find_min_max_percentile(temp, &vmin, &vmax);
		image = mve::image::raw_to_byte_image(temp, vmin, vmax);
		break;
	}

	case mve::IMAGE_TYPE_FLOAT:
	{
		mve::FloatImage::Ptr temp = mve::image::create_thumbnail<float>
			(std::dynamic_pointer_cast<mve::FloatImage const>(img),
				THUMBNAIL_SIZE, THUMBNAIL_SIZE);
		float vmin, vmax;
		find_min_max_percentile(temp, &vmin, &vmax);
		image = mve::image::float_to_byte_image(temp, vmin, vmax);
		break;
	}

	default:
		return mve::ByteImage::Ptr();
	}

	return image;
}

/* ---------------------------------------------------------------- */

std::string
remove_file_extension(std::string const& filename)
{
	std::size_t pos = filename.find_last_of('.');
	if (pos != std::string::npos)
		return filename.substr(0, pos);
	return filename;
}

/* ---------------------------------------------------------------- */

template <class T>
typename mve::Image<T>::Ptr
limit_image_size(typename mve::Image<T>::Ptr img, int max_pixels)
{
	while (img->get_pixel_amount() > max_pixels)
		img = mve::image::rescale_half_size<T>(img);
	return img;
}

/* ---------------------------------------------------------------- */

mve::ImageBase::Ptr
limit_image_size(mve::ImageBase::Ptr image, int max_pixels)
{
	switch (image->get_type())
	{
	case mve::IMAGE_TYPE_FLOAT:
		return limit_image_size<float>(std::dynamic_pointer_cast
			<mve::FloatImage>(image), max_pixels);
	case mve::IMAGE_TYPE_UINT8:
		return limit_image_size<uint8_t>(std::dynamic_pointer_cast
			<mve::ByteImage>(image), max_pixels);
	case mve::IMAGE_TYPE_UINT16:
		return limit_image_size<uint16_t>(std::dynamic_pointer_cast
			<mve::RawImage>(image), max_pixels);
	default:
		break;
	}
	return mve::ImageBase::Ptr();
}

/* ---------------------------------------------------------------- */

bool
has_jpeg_extension(std::string const& filename)
{
	std::string lcfname(util::string::lowercase(filename));
	return util::string::right(lcfname, 4) == ".jpg"
		|| util::string::right(lcfname, 5) == ".jpeg";
}

/* ---------------------------------------------------------------- */

std::string
make_image_name(int id)
{
	return "view_" + util::string::get_filled(id, 4) + ".mve";
}

/* ---------------------------------------------------------------- */

void
import_bundle_nvm(AppSettings const& conf)
{
	std::vector<mve::NVMCameraInfo> nvm_cams;
	mve::Bundle::Ptr bundle = mve::load_nvm_bundle(conf.path_image, &nvm_cams);
	mve::Bundle::Cameras& cameras = bundle->get_cameras();

	if (nvm_cams.size() != cameras.size())
	{
		std::cerr << "Error: NVM info inconsistent with bundle!" << std::endl;
		return;
	}

	/* Create output directories. */
	std::cout << "Creating output directories..." << std::endl;
	util::fs::mkdir(conf.path_scene.c_str());
	util::fs::mkdir(conf.path_views.c_str());

	/* Create and write views. */
	std::cout << "Writing MVE views..." << std::endl;
#pragma omp parallel for schedule(dynamic, 1)
#if !defined(_MSC_VER)
	for (std::size_t i = 0; i < cameras.size(); ++i)
#else
	for (int i = 0; i < cameras.size(); ++i)
#endif
	{
		mve::CameraInfo& mve_cam = cameras[i];
		mve::NVMCameraInfo const& nvm_cam = nvm_cams[i];
		std::string fname = "view_" + util::string::get_filled(i, 4) + ".mve";

		mve::View::Ptr view = mve::View::create();
		view->set_id(i);
		view->set_name(util::string::get_filled(i, 4, '0'));

		/* Load original image. */
		std::string exif;
		mve::ByteImage::Ptr image = load_8bit_image(nvm_cam.filename, &exif);
		if (image == nullptr)
		{
			std::cout << "Error loading: " << nvm_cam.filename
				<< " (skipping " << fname << ")" << std::endl;
			continue;
		}

		/* Add original image. */
		if (conf.import_orig)
		{
			if (has_jpeg_extension(nvm_cam.filename))
				view->set_image_ref(nvm_cam.filename, "original");
			else
				view->set_image(image, "original");
		}
		view->set_image(create_thumbnail(image), "thumbnail");
		add_exif_to_view(view, exif);

		/* Normalize focal length, add undistorted image. */
		int const maxdim = std::max(image->width(), image->height());
		mve_cam.flen = mve_cam.flen / static_cast<float>(maxdim);

		mve::ByteImage::Ptr undist = mve::image::image_undistort_vsfm<uint8_t>
			(image, mve_cam.flen, nvm_cam.radial_distortion);
		undist = limit_image_size<uint8_t>(undist, conf.max_pixels);
		view->set_image(undist, "undistorted");
		view->set_camera(mve_cam);

		/* Save view. */
#pragma omp critical
		std::cout << "Writing MVE view: " << fname << "..." << std::endl;
		view->save_view_as(util::fs::join_path(conf.path_views, fname));
	}

	/* Use MVE to write MVE bundle file. */
	std::cout << "Writing bundle file..." << std::endl;
	std::string bundle_filename
		= util::fs::join_path(conf.path_scene, "synth_0.out");
	mve::save_mve_bundle(bundle, bundle_filename);

	std::cout << std::endl << "Done importing NVM file!" << std::endl;
}

/* ---------------------------------------------------------------- */

namespace
{
	enum BundleFormat
	{
		BUNDLE_FORMAT_NOAH_BUNDLER,
		BUNDLE_FORMAT_PHOTOSYNTHER,
		BUNDLE_FORMAT_UNKNOWN
	};
}

void
import_bundle_noah_ps(AppSettings const& conf)
{
	/* Build some paths. */
	std::string bundle_fname;
	std::string imglist_file;
	std::string image_path;
	std::string undist_path;
	BundleFormat bundler_fmt = BUNDLE_FORMAT_UNKNOWN;
	bool import_original = conf.import_orig;

	/*
	* Try to detect Photosynther software. This is detected if the
	* file synth_N.out (for bundle N) is present in the bundler dir.
	*/
	if (bundler_fmt == BUNDLE_FORMAT_UNKNOWN)
	{
		bundle_fname = "synth_" + util::string::get(conf.bundle_id) + ".out";
		bundle_fname = util::fs::join_path(conf.path_bundle, bundle_fname);
		imglist_file = util::fs::join_path(conf.path_image, PS_BUNDLE_LOG);
		image_path = util::fs::join_path(conf.path_image, PS_IMAGE_DIR);
		undist_path = util::fs::join_path(conf.path_image, PS_UNDIST_DIR);

		if (util::fs::file_exists(bundle_fname.c_str()))
			bundler_fmt = BUNDLE_FORMAT_PHOTOSYNTHER;
	}

	/*
	* Try to detect Noah bundler software. Noah bundler is detected if
	* file bundle.out (for bundle 0) or bundle_%03d.out (for bundle > 0)
	* is present in the bundler directory.
	*/
	if (bundler_fmt == BUNDLE_FORMAT_UNKNOWN)
	{
		if (conf.bundle_id > 0)
			bundle_fname = "bundle_" + util::string::get_filled
			(conf.bundle_id, 3, '0') + ".out";
		else
			bundle_fname = "bundle.out";

		bundle_fname = util::fs::join_path(conf.path_bundle, bundle_fname);
		imglist_file = util::fs::join_path(conf.path_image, BUNDLER_FILE_LIST);
		image_path = util::fs::join_path(conf.path_image, BUNDLER_IMAGE_DIR);

		if (util::fs::file_exists(bundle_fname.c_str()))
			bundler_fmt = BUNDLE_FORMAT_NOAH_BUNDLER;
	}

	/* Read bundle file. */
	mve::Bundle::Ptr bundle;
	try
	{
		if (bundler_fmt == BUNDLE_FORMAT_NOAH_BUNDLER)
			bundle = mve::load_bundler_bundle(bundle_fname);
		else if (bundler_fmt == BUNDLE_FORMAT_PHOTOSYNTHER)
			bundle = mve::load_photosynther_bundle(bundle_fname);
		else
		{
			std::cerr << "Error: Could not detect bundle format." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "Error reading bundle: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	/* Read the list of original images filenames. */
	StringVector orig_files;
	if (bundler_fmt == BUNDLE_FORMAT_PHOTOSYNTHER && import_original)
	{
		std::cerr << std::endl << "** Warning: Original images cannot be "
			<< "imported from Photosynther." << std::endl;
		wait_for_user_confirmation();
		import_original = false;
	}
	else if (bundler_fmt == BUNDLE_FORMAT_NOAH_BUNDLER)
	{
		/*
		* Each camera in the bundle file corresponds to the ordered list of
		* input images. Some cameras are set to zero, which means the input
		* image was not registered. The paths of original images is required
		* because Bundler does not compute the undistorted images.
		*/
		read_noah_imagelist(imglist_file, orig_files);
		if (orig_files.empty())
		{
			std::cerr << "Error: Empty list of original images." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		if (orig_files.size() != bundle->get_num_cameras())
		{
			std::cerr << "Error: False amount of original images." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::cout << "Recognized " << orig_files.size()
			<< " original images from Noah's Bundler." << std::endl;
	}

	/* ------------------ Start importing views ------------------- */

	/* Create destination directories. */
	std::cout << "Creating output directories..." << std::endl;
	util::fs::mkdir(conf.path_scene.c_str());
	util::fs::mkdir(conf.path_views.c_str());

	/* Save bundle file. */
	std::cout << "Saving bundle file..." << std::endl;
	mve::save_photosynther_bundle(bundle,
		util::fs::join_path(conf.path_scene, "synth_0.out"));

	/* Save MVE views. */
	int num_valid_cams = 0;
	int undist_imported = 0;
	mve::Bundle::Cameras const& cams = bundle->get_cameras();
	for (std::size_t i = 0; i < cams.size(); ++i)
	{
		/*
		* For each camera in the bundle file, a new view is created.
		* Views are populated with ID, name, camera information,
		* undistorted RGB image, and optionally the original RGB image.
		*/
		std::string fname = "view_" + util::string::get_filled(i, 4) + ".mve";
		std::cout << "Processing view " << fname << "..." << std::endl;

		/* Skip invalid cameras... */
		mve::CameraInfo cam = cams[i];
		if (cam.flen == 0.0f && (conf.skip_invalid
			|| bundler_fmt == BUNDLE_FORMAT_PHOTOSYNTHER))
		{
			std::cerr << "  Skipping " << fname
				<< ": Invalid camera." << std::endl;
			continue;
		}

		/* Extract name of view from original image or sequentially. */
		std::string view_name = (import_original
			? remove_file_extension(util::fs::basename(orig_files[i]))
			: util::string::get_filled(i, 4, '0'));

		/* Convert from Photosynther camera conventions. */
		if (bundler_fmt == BUNDLE_FORMAT_PHOTOSYNTHER)
		{
			/* Nothing to do here. */
		}

		/* Fix issues with Noah Bundler camera specification. */
		if (bundler_fmt == BUNDLE_FORMAT_NOAH_BUNDLER)
		{
			/* Check focal length of camera, fix negative focal length. */
			if (cam.flen < 0.0f)
			{
				std::cout << "  Fixing focal length for " << fname << std::endl;
				cam.flen = -cam.flen;
				std::for_each(cam.rot, cam.rot + 9,
					math::algo::foreach_negate_value<float>);
				std::for_each(cam.trans, cam.trans + 3,
					math::algo::foreach_negate_value<float>);
			}

			/* Convert from Noah Bundler camera conventions. */
			std::for_each(cam.rot + 3, cam.rot + 9,
				math::algo::foreach_negate_value<float>);
			std::for_each(cam.trans + 1, cam.trans + 3,
				math::algo::foreach_negate_value<float>);

			/* Check determinant of rotation matrix. */
			math::Matrix3f rmat(cam.rot);
			float rmatdet = math::matrix_determinant(rmat);
			if (rmatdet < 0.0f)
			{
				std::cerr << "  Skipping " << fname
					<< ": Bad rotation matrix." << std::endl;
				continue;
			}
		}

		/* Create view and set headers. */
		mve::View::Ptr view = mve::View::create();
		view->set_id(i);
		view->set_name(view_name);
		view->set_camera(cam);

		/* FIXME: Handle exceptions while loading images? */

		/* Load undistorted and original image, create thumbnail. */
		mve::ByteImage::Ptr original, undist, thumb;
		std::string exif;
		if (bundler_fmt == BUNDLE_FORMAT_NOAH_BUNDLER)
		{
			/* For Noah datasets, load original image and undistort it. */
			std::string orig_fname
				= util::fs::join_path(image_path, orig_files[i]);
			original = load_8bit_image(orig_fname, &exif);
			if (original == nullptr)
			{
				std::cerr << "Error loading: " << orig_fname << std::endl;
				std::exit(EXIT_FAILURE);
			}
			thumb = create_thumbnail(original);

			/* Convert Bundler's focal length to MVE focal length. */
			cam.flen /= (float)std::max(original->width(), original->height());
			view->set_camera(cam);

			if (cam.flen != 0.0f)
				undist = mve::image::image_undistort_k2k4<uint8_t>
				(original, cam.flen, cam.dist[0], cam.dist[1]);

			if (!import_original)
				original.reset();
		}
		else if (bundler_fmt == BUNDLE_FORMAT_PHOTOSYNTHER)
		{
			/*
			* Depending on the version, try to load two file names:
			* New version: forStereo_xxxx_yyyy.png
			* Old version: undistorted_xxxx_yyyy.jpg
			*/
			std::string undist_new_filename
				= util::fs::join_path(undist_path, "forStereo_"
					+ util::string::get_filled(conf.bundle_id, 4) + "_"
					+ util::string::get_filled(num_valid_cams, 4) + ".png");
			std::string undist_old_filename
				= util::fs::join_path(undist_path, "undistorted_"
					+ util::string::get_filled(conf.bundle_id, 4) + "_"
					+ util::string::get_filled(num_valid_cams, 4) + ".jpg");

			/* Try the newer file name and fall back if not existing. */
			try
			{
				if (util::fs::file_exists(undist_new_filename.c_str()))
					undist = mve::image::load_file(undist_new_filename);
				else
					undist = mve::image::load_file(undist_old_filename);
			}
			catch (util::FileException &e)
			{
				std::cerr << e.filename << ": " << e.what() << std::endl;
				std::exit(EXIT_FAILURE);
			}
			catch (util::Exception &e)
			{
				std::cerr << e.what() << std::endl;
				std::exit(EXIT_FAILURE);
			}

			/* Create thumbnail. */
			thumb = create_thumbnail(undist);
		}

		/* Add images to view. */
		if (thumb != nullptr)
			view->set_image(thumb, "thumbnail");

		if (undist != nullptr)
		{
			undist = limit_image_size<uint8_t>(undist, conf.max_pixels);
			view->set_image(undist, "undistorted");
		}
		else if (cam.flen != 0.0f && undist == nullptr)
			std::cerr << "Warning: Undistorted image missing!" << std::endl;

		if (original != nullptr)
			view->set_image(original, "original");
		if (original == nullptr && import_original)
			std::cerr << "Warning: Original image missing!" << std::endl;

		/* Add EXIF data to view if available. */
		add_exif_to_view(view, exif);

		/* Save MVE file. */
		view->save_view_as(util::fs::join_path(conf.path_views, fname));

		if (cam.flen != 0.0f)
			num_valid_cams += 1;
		if (undist != nullptr)
			undist_imported += 1;
	}

	std::cout << std::endl;
	std::cout << "Created " << cams.size() << " views with "
		<< num_valid_cams << " valid cameras." << std::endl;
	std::cout << "Imported " << undist_imported
		<< " undistorted images." << std::endl;
}

/* ---------------------------------------------------------------- */

bool
is_visual_sfm_bundle_format(AppSettings const& conf)
{
	return util::string::right(conf.path_image, 4) == ".nvm" && util::fs::file_exists(conf.path_image.c_str());
}

bool
is_photosynther_bundle_format(AppSettings const& conf)
{
	std::string bundle_fname = util::fs::join_path(conf.path_bundle,
		"synth_" + util::string::get(conf.bundle_id) + ".out");
	return util::fs::file_exists(bundle_fname.c_str());
}

bool
is_noah_bundler_format(AppSettings const& conf)
{
	std::string bundle_fname = util::fs::join_path(conf.path_bundle,
		conf.bundle_id == 0 ? "bundle.out" : "bundle_"
		+ util::string::get_filled(conf.bundle_id, 3, '0') + ".out");
	return util::fs::file_exists(bundle_fname.c_str());
}

/* ---------------------------------------------------------------- */

void import_bundle(AppSettings const& conf)
{
	/**
	* Try to detect VisualSFM bundle format.
	* In this case the input is a file with extension ".nvm".
	*/
	if (is_visual_sfm_bundle_format(conf))
	{
		std::cout << "Info: Detected VisualSFM bundle format." << std::endl;
		import_bundle_nvm(conf);
		return;
	}

	/**
	* Try to detect Noah bundler or Photosynther. These bundle formats
	* are similar and handled with the same import function.
	*/
	if (is_photosynther_bundle_format(conf))
	{
		std::cout << "Info: Detected Photosynther bundle format." << std::endl;
		import_bundle_noah_ps(conf);
		return;
	}
	if (is_noah_bundler_format(conf))
	{
		std::cout << "Info: Detected Noah bundler format." << std::endl;
		import_bundle_noah_ps(conf);
		return;
	}
}

/* ---------------------------------------------------------------- */

int
find_max_scene_id(std::string const& view_path)
{
	util::fs::Directory dir;
	try { dir.scan(view_path); }
	catch (...) { return -1; }

	/* Load all MVE files and remember largest view ID. */
	int max_view_id = 0;
	for (std::size_t i = 0; i < dir.size(); ++i)
	{
		std::string ext4 = util::string::right(dir[i].name, 4);
		if (ext4 != ".mve")
			continue;

		mve::View::Ptr view;
		try
		{
			view = mve::View::create(dir[i].get_absolute_name());
		}
		catch (...)
		{
			std::cerr << "Error reading " << dir[i].name << std::endl;
			continue;
		}

		max_view_id = std::max(max_view_id, view->get_id());
	}

	return max_view_id;
}

/* ---------------------------------------------------------------- */

void import_images(AppSettings const& conf)
{
	util::WallTimer timer;

	util::fs::Directory dir;
	try { dir.scan(conf.path_image); }
	catch (std::exception& e)
	{
		std::cerr << "Error scanning input dir: " << e.what() << std::endl;
		log_message(conf, "Error scanning input dir: ");
		log_message(conf, e.what());

		std::exit(EXIT_FAILURE);
	}
	std::cout << "Found " << dir.size() << " directory entries." << std::endl;

	/* ------------------ Start importing images ------------------- */

	/* Create destination dir. */
	if (!conf.append_images)
	{
		std::cout << "Creating output directories..." << std::endl;
		util::fs::mkdir(conf.path_scene.c_str());
		util::fs::mkdir(conf.path_views.c_str());
	}

	int max_scene_id = -1;
	if (conf.append_images)
	{
		max_scene_id = find_max_scene_id(conf.path_views);
		if (max_scene_id < 0)
		{
// 			std::cerr << "Error: Cannot find view ID for appending." << std::endl;
			log_message(conf, "Error: Cannot find view ID for appending.");

			std::exit(EXIT_FAILURE);
		}
	}

	/* Sort file names, iterate over file names. */
	std::sort(dir.begin(), dir.end());
	std::atomic_int id_cnt(max_scene_id + 1);
	std::atomic_int num_imported(0);
#pragma omp parallel for ordered schedule(dynamic,1)
#if !defined(_MSC_VER)
	for (std::size_t i = 0; i < dir.size(); ++i)
#else
	for (int64_t i = 0; i < dir.size(); ++i)
#endif
	{
		if (dir[i].is_dir)
		{
#pragma omp critical
			std::cout << "Skipping directory " << dir[i].name << std::endl;
			continue;
		}

		std::string fname = dir[i].name;
		std::string afname = dir[i].get_absolute_name();

		std::string exif;
		mve::ImageBase::Ptr image = load_any_image(afname, &exif);
		if (image == nullptr)
			continue;

		/* Advance ID of successfully imported images. */
		int id;
#pragma omp ordered
		id = id_cnt++;
		num_imported += 1;

		/* Create view, set headers, add image. */
		mve::View::Ptr view = mve::View::create();
		view->set_id(id);
		view->set_name(remove_file_extension(fname));

		/* Rescale and add original image. */
		int orig_width = image->width();
		image = limit_image_size(image, conf.max_pixels);
		if (orig_width == image->width() && has_jpeg_extension(fname))
		{
			view->set_image_ref(afname, "original");
			img_extension = "jpg";
		}
		else
			view->set_image(image, "original");

		/* Add thumbnail for byte images. */
		mve::ByteImage::Ptr thumb = create_thumbnail(image);
		if (thumb != nullptr)
			view->set_image(thumb, "thumbnail");

		/* Add EXIF data to view if available. */
		add_exif_to_view(view, exif);

		/* Save view to disc. */
		std::string mve_fname = make_image_name(id);
#pragma omp critical
		std::cout << "Importing image: " << fname
			<< ", writing MVE view: " << mve_fname << "..." << std::endl;
		view->save_view_as(util::fs::join_path(conf.path_views, mve_fname));
	}

	std::cout << "Imported " << num_imported << " input images, " << "took " << timer.get_elapsed() << " ms." << std::endl;
}

/* ---------------------------------------------------------------- */

void log_message(AppSettings const& conf, std::string const& message)
{
	std::cout << message << std::endl;

	if (conf.log_file.empty())
		return;

	std::string fname = util::fs::join_path(conf.path_scene, conf.log_file);
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

void compute_feature(mve::Scene::Ptr scene, sfm::bundler::ViewportList* viewports)
{
	if (scene == nullptr)
		throw std::invalid_argument("Null scene given");
	if (viewports == nullptr)
		throw std::invalid_argument("No viewports given");

	mve::Scene::ViewList const& views = scene->get_views();

	/* Initialize viewports. */
	viewports->clear();
	viewports->resize(views.size());

	std::size_t num_views = viewports->size();
	std::size_t num_done = 0;
	std::size_t total_features = 0;

	/* Iterate the scene and compute features. */
	// #pragma omp parallel for schedule(dynamic,1)
#ifdef _MSC_VER
	for (int64_t i = 0; i < views.size(); ++i)
#else
	for (std::size_t i = 0; i < views.size(); ++i)
#endif
	{
		// #pragma omp critical
		{
			num_done += 1;
			float percent = (num_done * 1000 / num_views) / 10.0f;
			std::cout << "\rDetecting features, view " << num_done << " of " << num_views << " (" << percent << "%)..." << std::flush;
		}

		if (views[i] == nullptr)
			continue;

		mve::View::Ptr view = views[i];
		std::string directory = view->get_directory();
		std::string filename = util::fs::join_path(directory, "original.jpg");

		Ptr<Feature2D> b;
		b = AKAZE::create();

#if 0
		try
		{
			// We can detect keypoint with detect method
			b->detect(img1, keyImg1, Mat());
			// and compute their descriptors with method  compute
			b->compute(img1, keyImg1, descImg1);
			// or detect and compute descriptors in one step
			b->detectAndCompute(img2, Mat(), keyImg2, descImg2, false);
			// Match method loop
		}
		catch (const Exception& e)
		{
			cerr << "Exception: " << e.what() << endl;
			cout << "Feature : " << *itDesc << "\n";
		}



		mve::ByteImage::Ptr image = view->get_byte_image(this->opts.image_embedding);
		if (image == nullptr)
			continue;

		/* Rescale image until maximum image size is met. */
		util::WallTimer timer;
		while (this->opts.max_image_size > 0 && image->width() * image->height() > this->opts.max_image_size)
			image = mve::image::rescale_half_size<uint8_t>(image);

		/* Compute features for view. */
		Viewport* viewport = &viewports->at(i);
		viewport->features.set_options(this->opts.feature_options);
		viewport->features.compute_features(image);
		std::size_t num_feats = viewport->features.positions.size();

		/* Normalize image coordinates. */
		float const fwidth = static_cast<float>(viewport->features.width);
		float const fheight = static_cast<float>(viewport->features.height);
		float const fnorm = std::max(fwidth, fheight);
		for (std::size_t j = 0; j < viewport->features.positions.size(); ++j)
		{
			math::Vec2f& pos = viewport->features.positions[j];
			pos[0] = (pos[0] + 0.5f - fwidth / 2.0f) / fnorm;
			pos[1] = (pos[1] + 0.5f - fheight / 2.0f) / fnorm;
		}

		// #pragma omp critical
		{
			std::cout << "\rView ID "
				<< util::string::get_filled(view->get_id(), 4, '0') << " ("
				<< image->width() << "x" << image->height() << "), "
				<< util::string::get_filled(num_feats, 5, ' ') << " features"
				<< ", took " << timer.get_elapsed() << " ms." << std::endl;
			total_features += viewport->features.positions.size();
		}

		/* Clean up unused embeddings. */
		image.reset();
#endif
		view->cache_cleanup();
	}

	std::cout << "\rComputed " << total_features << " features " << "for " << num_views << " views (average " << (total_features / num_views) << ")." << std::endl;
}

void features_and_matching(mve::Scene::Ptr scene, AppSettings const& conf, sfm::bundler::ViewportList* viewports, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	/* Feature computation for the scene. */
	sfm::bundler::Features::Options feature_opts;
	feature_opts.image_embedding = conf.original_name;
	feature_opts.max_image_size = conf.max_image_size;
	feature_opts.feature_options.feature_types = sfm::FeatureSet::FEATURE_SIFT;//sfm::FeatureSet::FEATURE_SIFT;//sfm::FeatureSet::FEATURE_ALL;

	std::cout << "Computing image features..." << std::endl;
	{
		util::WallTimer timer;
		sfm::bundler::Features bundler_features(feature_opts);

// 		compute_feature(scene, viewports);
		bundler_features.compute(scene, viewports);

		std::cout << "Computing features took " << timer.get_elapsed() << " ms." << std::endl;
		log_message(conf, "Feature detection took " + util::string::get(timer.get_elapsed()) + "ms.");
	}

	/* Exhaustive matching between all pairs of views. */
	sfm::bundler::Matching::Options matching_opts;
	//matching_opts.ransac_opts.max_iterations = 1000;
	//matching_opts.ransac_opts.threshold = 0.0015;
	matching_opts.ransac_opts.verbose_output = false;
	matching_opts.use_lowres_matching = conf.lowres_matching;
	matching_opts.match_num_previous_frames = conf.video_matching;
	matching_opts.matcher_type = conf.cascade_hashing
		? sfm::bundler::Matching::MATCHER_CASCADE_HASHING
		: sfm::bundler::Matching::MATCHER_EXHAUSTIVE;

	std::cout << "Performing feature matching..." << std::endl;
	{
		util::WallTimer timer;
		sfm::bundler::Matching bundler_matching(matching_opts);
		bundler_matching.init(viewports);
		for (int i = 0; i < viewports->size(); ++i)
		{
			std::cout << "feature count per viewport:" << viewports->at(i).features.positions.size() << std::endl;
		}
		bundler_matching.compute(pairwise_matching);
		std::cout << "Matching took " << timer.get_elapsed() << " ms." << std::endl;
		log_message(conf, "Feature matching took " + util::string::get(timer.get_elapsed()) + "ms.");
	}

	if (pairwise_matching->empty()) {
		std::cerr << "Error: No matching image pairs. Exiting." << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

void cv_features_matching(mve::Scene::Ptr scene, AppSettings const& conf, std::string &img_extension, sfm::bundler::ViewportList* viewports,
							 sfm::bundler::PairwiseMatching* pairwise_matching)
{
	sfm::bundler::Matching::Options matching_opts;
	//matching_opts.ransac_opts.max_iterations = 1000;
	//matching_opts.ransac_opts.threshold = 0.0015;
	matching_opts.ransac_opts.verbose_output = false;
	matching_opts.use_lowres_matching = conf.lowres_matching;
	matching_opts.match_num_previous_frames = conf.video_matching;
	matching_opts.matcher_type = conf.cascade_hashing
		? sfm::bundler::Matching::MATCHER_CASCADE_HASHING
		: sfm::bundler::Matching::MATCHER_EXHAUSTIVE;
	sfm::bundler::Matching bundler_matching(matching_opts);
	cv_multi_feature_matching(scene, conf, img_extension, bundler_matching, viewports, pairwise_matching);
	//cv_features_and_matching(scene, conf, bundler_matching, viewports, pairwise_matching);
// 	combine_akaze_matches(scene, bundler_matching, viewports, pairwise_matching);
}

void sfm_reconstruct(AppSettings const& conf)
{
#if ENABLE_SSE2_NN_SEARCH && defined(__SSE2__)
	std::cout << "SSE2 accelerated matching is enabled." << std::endl;
#else
	std::cout << "SSE2 accelerated matching is disabled." << std::endl;
#endif

#if ENABLE_SSE3_NN_SEARCH && defined(__SSE3__)
	std::cout << "SSE3 accelerated matching is enabled." << std::endl;
#else
	std::cout << "SSE3 accelerated matching is disabled." << std::endl;
#endif

	/* Load scene. */
	mve::Scene::Ptr scene;
	try {
		scene = mve::Scene::create(conf.path_scene);
	} catch (std::exception& e) {
		std::cerr << "Error loading scene: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	sfm::bundler::SurveyPointList survey;
	if (!conf.survey_file.empty())
		sfm::bundler::load_survey_from_file(conf.survey_file, &survey);

	/* Try to load the pairwise matching from the prebundle. */
	std::string const prebundle_path = util::fs::join_path(scene->get_path(), conf.prebundle_file);
	sfm::bundler::ViewportList viewports;
	sfm::bundler::PairwiseMatching pairwise_matching;

	if (!util::fs::file_exists(prebundle_path.c_str())) {
		log_message(conf, "Starting feature matching.");
		util::system::rand_seed(RAND_SEED_MATCHING);
 		features_and_matching(scene, conf, &viewports, &pairwise_matching);
		//cv_features_matching(scene, conf, img_extension, &viewports, &pairwise_matching);
		
// 		std::cout << "Saving pre-bundle to file..." << std::endl;
		log_message(conf, "Saving pre-bundle to file...");

		sfm::bundler::save_prebundle_to_file(viewports, pairwise_matching, prebundle_path);
	} else if (!conf.skip_sfm) {
// 		std::cout << "Loading pairwise matching from file..." << std::endl;
		log_message(conf, "Loading pairwise matching from file...");
		sfm::bundler::load_prebundle_from_file(prebundle_path, &viewports, &pairwise_matching);
	}

	if (conf.skip_sfm) {
// 		std::cout << "Pre-bundle finished, skipping SfM. Exiting." << std::endl;
		log_message(conf, "Pre-bundle finished, skipping SfM. Exiting.");
		std::exit(EXIT_SUCCESS);
	}

	/* Drop descriptors and embeddings to save memory. */
	scene->cache_cleanup();
	for (std::size_t i = 0; i < viewports.size(); ++i)
		viewports[i].features.clear_descriptors();

	/* Check if there are some matching images. */
	if (pairwise_matching.empty())
	{
// 		std::cerr << "No matching image pairs. Exiting." << std::endl;
		log_message(conf, "No matching image pairs. Exiting.");
		std::exit(EXIT_FAILURE);
	}

	/*
	* Obtain camera intrinsics from the views or guess them from EXIF.
	* If neither is available, fall back to a default.
	*
	* FIXME: Once width and height in the viewport is gone, this fully
	* initializes the viewports. Thus viewport info does not need to be
	* saved in the prebundle.sfm, and the file becomes matching.sfm.
	* Obtaining EXIF guesses can be moved out of the feature module to here.
	* The following code can become its own module "bundler_intrinsics".
	*/
	{
		sfm::bundler::Intrinsics::Options intrinsics_opts;
		if (conf.intrinsics_from_views)
		{
			intrinsics_opts.intrinsics_source
				= sfm::bundler::Intrinsics::FROM_VIEWS;
		}
// 		std::cout << "Initializing camera intrinsics..." << std::endl;
		log_message(conf, "Initializing camera intrinsics...");
		sfm::bundler::Intrinsics intrinsics(intrinsics_opts);
		intrinsics.compute(scene, &viewports);
	}

	/* Start incremental SfM. */
	log_message(conf, "Starting incremental SfM.");
	util::WallTimer timer;
	util::system::rand_seed(RAND_SEED_SFM);

	/* Compute connected feature components, i.e. feature tracks. */
	sfm::bundler::TrackList tracks;
	{
		sfm::bundler::Tracks::Options tracks_options;
		tracks_options.verbose_output = true;

		sfm::bundler::Tracks bundler_tracks(tracks_options);
// 		std::cout << "Computing feature tracks..." << std::endl;
		log_message(conf, "Computing feature tracks...");

		bundler_tracks.compute(pairwise_matching, &viewports, &tracks);
		std::cout << "Created a total of " << tracks.size() << " tracks." << std::endl;
	}

	/* Remove color data and pairwise matching to save memory. */
	for (std::size_t i = 0; i < viewports.size(); ++i)
		viewports[i].features.colors.clear();
	pairwise_matching.clear();

	/* Search for a good initial pair, or use the user-specified one. */
	sfm::bundler::InitialPair::Result init_pair_result;
	sfm::bundler::InitialPair::Options init_pair_opts;
	if (conf.initial_pair_1 < 0 || conf.initial_pair_2 < 0)
	{
		//init_pair_opts.homography_opts.max_iterations = 1000;
		//init_pair_opts.homography_opts.threshold = 0.005f;
		init_pair_opts.homography_opts.verbose_output = false;
		init_pair_opts.max_homography_inliers = 0.8f;
		init_pair_opts.verbose_output = true;

		sfm::bundler::InitialPair init_pair(init_pair_opts);
		init_pair.initialize(viewports, tracks);
		init_pair.compute_pair(&init_pair_result);
	}
	else
	{
// 		std::cout << "Reconstructing initial pair..." << std::endl;
		log_message(conf, "Reconstructing initial pair...");
		sfm::bundler::InitialPair init_pair(init_pair_opts);
		init_pair.initialize(viewports, tracks);
		init_pair.compute_pair(conf.initial_pair_1, conf.initial_pair_2,
			&init_pair_result);
	}

	if (init_pair_result.view_1_id < 0 || init_pair_result.view_2_id < 0
		|| init_pair_result.view_1_id >= static_cast<int>(viewports.size())
		|| init_pair_result.view_2_id >= static_cast<int>(viewports.size()))
	{
// 		std::cerr << "Error finding initial pair, exiting!" << std::endl;
// 		std::cerr << "Try manually specifying an initial pair." << std::endl;
		log_message(conf, "Error finding initial pair, exiting!");
		log_message(conf, "Try manually specifying an initial pair.");
		std::exit(EXIT_FAILURE);
	}

	std::cout << "Using views " << init_pair_result.view_1_id
		<< " and " << init_pair_result.view_2_id
		<< " as initial pair." << std::endl;

	/* Incrementally compute full bundle. */
	sfm::bundler::Incremental::Options incremental_opts;
	//incremental_opts.pose_p3p_opts.max_iterations = 1000;
	//incremental_opts.pose_p3p_opts.threshold = 0.005f;
	incremental_opts.pose_p3p_opts.verbose_output = false;
	incremental_opts.track_error_threshold_factor = conf.track_error_thres_factor;
	incremental_opts.new_track_error_threshold = conf.new_track_error_thres;
	incremental_opts.min_triangulation_angle = MATH_DEG2RAD(1.0);
	incremental_opts.ba_fixed_intrinsics = conf.fixed_intrinsics;
	//incremental_opts.ba_shared_intrinsics = conf.shared_intrinsics;
	incremental_opts.verbose_output = true;
	incremental_opts.verbose_ba = conf.verbose_ba;

	/* Initialize viewports with initial pair. */
	viewports[init_pair_result.view_1_id].pose = init_pair_result.view_1_pose;
	viewports[init_pair_result.view_2_id].pose = init_pair_result.view_2_pose;

	/* Initialize the incremental bundler and reconstruct first tracks. */
	sfm::bundler::Incremental incremental(incremental_opts);
	incremental.initialize(&viewports, &tracks, &survey);
	incremental.triangulate_new_tracks(2);
	incremental.invalidate_large_error_tracks();

	/* Run bundle adjustment. */
	std::cout << "Running full bundle adjustment..." << std::endl;
	incremental.bundle_adjustment_full();

	/* Reconstruct remaining views. */
	int num_cameras_reconstructed = 2;
	int full_ba_num_skipped = 0;
	while (true)
	{
		/* Find suitable next views for reconstruction. */
		std::vector<int> next_views;
		incremental.find_next_views(&next_views);

		/* Reconstruct the next view. */
		int next_view_id = -1;
		for (std::size_t i = 0; i < next_views.size(); ++i)
		{
			std::cout << std::endl;
			std::cout << "Adding next view ID " << next_views[i]
				<< " (" << (num_cameras_reconstructed + 1) << " of "
				<< viewports.size() << ")..." << std::endl;
			if (incremental.reconstruct_next_view(next_views[i]))
			{
				next_view_id = next_views[i];
				break;
			}
		}

		if (next_view_id < 0)
		{
			if (full_ba_num_skipped == 0)
			{
				std::cout << "No valid next view." << std::endl;
				std::cout << "SfM reconstruction finished." << std::endl;
				break;
			}
			else
			{
				incremental.triangulate_new_tracks(conf.min_views_per_track);
				std::cout << "Running full bundle adjustment..." << std::endl;
				incremental.bundle_adjustment_full();
				incremental.invalidate_large_error_tracks();
				full_ba_num_skipped = 0;
				continue;
			}
		}

		/* Run single-camera bundle adjustment. */
		std::cout << "Running single camera bundle adjustment..." << std::endl;
		incremental.bundle_adjustment_single_cam(next_view_id);
		num_cameras_reconstructed += 1;

		/* Run full bundle adjustment only after a couple of views. */
		int const full_ba_skip_views = conf.always_full_ba ? 0
			: std::min(100, num_cameras_reconstructed / 10);
		if (full_ba_num_skipped < full_ba_skip_views)
		{
			std::cout << "Skipping full bundle adjustment (skipping "
				<< full_ba_skip_views << " views)." << std::endl;
			full_ba_num_skipped += 1;
		}
		else
		{
			incremental.triangulate_new_tracks(conf.min_views_per_track);
			std::cout << "Running full bundle adjustment..." << std::endl;
			incremental.bundle_adjustment_full();
			incremental.invalidate_large_error_tracks();
			full_ba_num_skipped = 0;
		}
	}

	std::cout << "SfM reconstruction took " << timer.get_elapsed() << " ms." << std::endl;
	log_message(conf, "SfM reconstruction took " + util::string::get(timer.get_elapsed()) + "ms.");

	/* Normalize scene if requested. */
	if (conf.normalize_scene)
	{
		std::cout << "Normalizing scene..." << std::endl;
		incremental.normalize_scene();
	}

	/* Save bundle file to scene. */
	std::cout << "Creating bundle data structure..." << std::endl;
	mve::Bundle::Ptr bundle = incremental.create_bundle();
	mve::save_mve_bundle(bundle, scene->get_path() + "/synth_0.out");

	/* Apply bundle cameras to views. */
	mve::Bundle::Cameras const& bundle_cams = bundle->get_cameras();
	mve::Scene::ViewList const& views = scene->get_views();
	if (bundle_cams.size() != views.size())
	{
		std::cerr << "Error: Invalid number of cameras!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

#pragma omp parallel for schedule(dynamic,1)
#ifndef _MSC_VER
	for (std::size_t i = 0; i < bundle_cams.size(); ++i)
#else
	for (int64_t i = 0; i < bundle_cams.size(); ++i)
#endif
	{
		mve::View::Ptr view = views[i];
		mve::CameraInfo const& cam = bundle_cams[i];
		if (view == nullptr)
			continue;
		if (view->get_camera().flen == 0.0f && cam.flen == 0.0f)
			continue;

		view->set_camera(cam);

		/* Undistort image. */
		if (!conf.undistorted_name.empty())
		{
			mve::ByteImage::Ptr original
				= view->get_byte_image(conf.original_name);
			if (original == nullptr)
				continue;
			mve::ByteImage::Ptr undist
				= mve::image::image_undistort_k2k4<uint8_t>
				(original, cam.flen, cam.dist[0], cam.dist[1]);
			view->set_image(undist, conf.undistorted_name);
		}

#pragma omp critical
		std::cout << "Saving view " << view->get_directory() << std::endl;
		view->save_view();
		view->cache_cleanup();
	}

	log_message(conf, "SfM reconstruction done.\n");
}

void check_prebundle(AppSettings const& conf)
{
	std::string const prebundle_path = util::fs::join_path(conf.path_scene, conf.prebundle_file);

	if (util::fs::exists(prebundle_path.c_str()))
		return;

	/* Check if the prebundle is writable. */
	std::ofstream out(prebundle_path.c_str(), std::ios::binary);
	if (!out.good()) {
		out.close();
		std::cerr << "Error: Specified prebundle not writable: " << prebundle_path << std::endl;
		std::cerr << "Note: The prebundle is relative to the scene." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	out.close();

	/* Looks good. Delete created prebundle. */
	util::fs::unlink(prebundle_path.c_str());
}

int make_scene(AppSettings& conf)
{
	log_message(conf, "Make scene starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("MVE Makescene");

	conf.path_views = util::fs::join_path(conf.path_scene, VIEWS_DIR);
	conf.path_bundle = util::fs::join_path(conf.path_image, BUNDLE_PATH);

	/* General settings. */
	conf.import_orig = true;
	conf.skip_invalid = false;
	conf.images_only = true;
	conf.append_images = false;
	conf.max_pixels = 6000000;//added at 01/06/2022
	/* Check command line arguments. */
	if (conf.path_image.empty() || conf.path_scene.empty()) {
		log_message(conf, "Image path or Scene path is empty.");

		return EXIT_FAILURE;
	}

	if (conf.append_images && !conf.images_only) {
// 		std::cerr << "Error: Cannot --append-images without --images-only." << std::endl;
		log_message(conf, "Error: Cannot --append-images without --images-only.");

		return EXIT_FAILURE;
	}

	/* Check if output dir exists. */
	bool output_path_exists = util::fs::dir_exists(conf.path_scene.c_str());
	if (output_path_exists && !conf.append_images) {
// 		std::cerr << std::endl;
// 		std::cerr << "** Warning: Output dir already exists." << std::endl;
// 		std::cerr << "** This may leave old views in your scene." << std::endl;
		//wait_for_user_confirmation();

		log_message(conf, "** Warning: Output dir already exists.");
		log_message(conf, "** This may leave old views in your scene.");
	}
	else if (!output_path_exists && conf.append_images) {
		std::cerr << "Error: Output dir does not exist. Cannot append images." << std::endl;
		log_message(conf, "Error: Output dir does not exist. Cannot append images.");

		return EXIT_FAILURE;
	}

	if (conf.images_only) {
		import_images(conf);
	} else {
		import_bundle(conf);
	}

	log_message(conf, "Make scene ends.");

	return EXIT_SUCCESS;
}

int sfmrecon(AppSettings& conf)
{
	log_message(conf, "SfM reconstruction starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("MVE SfM Reconstruction");

	try {
		check_prebundle(conf);
		sfm_reconstruct(conf);
	} catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		log_message(conf, e.what());
		std::exit(EXIT_FAILURE);
	}

	log_message(conf, "SfM reconstruction ends.");

	return EXIT_SUCCESS;
}

void generate_mesh(AppSettings& conf, mve::Scene::Ptr scene,
	std::string const& input_name, std::string const& dm_name)
{
	std::cout << "Generating ";
	if (conf.create_triangle_mesh)
		std::cout << "Mesh";
	else
		std::cout << "Pointcloud";
	if (conf.cut_surface)
		std::cout << ", Cutting surfaces";

	util::WallTimer timer;
	mve::Scene::ViewList recon_views;
	for (int i : conf.view_ids)
		recon_views.push_back(scene->get_views()[i]);

	std::cout << " for " << recon_views.size() << " views ..." << std::endl;

	smvs::MeshGenerator::Options meshgen_opts;
	meshgen_opts.num_threads = conf.num_threads;
	meshgen_opts.cut_surfaces = conf.cut_surface;
	meshgen_opts.simplify = conf.simplify;
	meshgen_opts.create_triangle_mesh = conf.create_triangle_mesh;

	smvs::MeshGenerator meshgen(meshgen_opts);
	mve::TriangleMesh::Ptr mesh = meshgen.generate_mesh(recon_views,
		input_name, dm_name);

	if (conf.aabb_string.size() > 0)
	{
		std::cout << "Clipping to AABB: (" << conf.aabb_min << ") / ("
			<< conf.aabb_max << ")" << std::endl;

		mve::TriangleMesh::VertexList const& verts = mesh->get_vertices();
		std::vector<bool> aabb_clip(verts.size(), false);
		for (std::size_t v = 0; v < verts.size(); ++v)
			for (int i = 0; i < 3; ++i)
				if (verts[v][i] < conf.aabb_min[i]
					|| verts[v][i] > conf.aabb_max[i])
					aabb_clip[v] = true;
		mesh->delete_vertices_fix_faces(aabb_clip);
	}

	std::cout << "Done. Took: " << timer.get_elapsed_sec() << "s" << std::endl;

	if (conf.create_triangle_mesh)
		mesh->recalc_normals();

	/* Build mesh name */
	std::string meshname = "smvs-";
	if (conf.create_triangle_mesh)
		meshname += "m-";
	if (conf.use_shading)
		meshname += "S";
	else
		meshname += "B";
	meshname += util::string::get(conf.input_scale) + ".ply";
	conf.dmname = meshname;
	conf.pset_name = util::fs::join_path(conf.path_scene, conf.dmname);

	meshname = util::fs::join_path(scene->get_path(), meshname);

	/* Save mesh */
	mve::geom::SavePLYOptions opts;
	opts.write_vertex_normals = true;
	opts.write_vertex_values = true;
	opts.write_vertex_confidences = true;
	mve::geom::save_ply_mesh(mesh, meshname, opts);
}

/* -------------------------------------------------------------------------- */

void reconstruct_sgm_depth_for_view(AppSettings const& conf,
	smvs::StereoView::Ptr main_view,
	std::vector<smvs::StereoView::Ptr> neighbors,
	mve::Bundle::ConstPtr bundle = nullptr)
{
	smvs::SGMStereo::Options sgm_opts;
	sgm_opts.scale = conf.sgm_scale;
	sgm_opts.num_steps = 128;
	sgm_opts.debug_lvl = conf.debug_lvl;
	sgm_opts.min_depth = conf.sgm_min;
	sgm_opts.max_depth = conf.sgm_max;

	util::WallTimer sgm_timer;
	mve::FloatImage::Ptr d1 = smvs::SGMStereo::reconstruct(sgm_opts, main_view,
		neighbors[0], bundle);
	if (neighbors.size() > 1)
	{
		mve::FloatImage::Ptr d2 = smvs::SGMStereo::reconstruct(sgm_opts,
			main_view, neighbors[1], bundle);
		for (int p = 0; p < d1->get_pixel_amount(); ++p)
		{
			if (d2->at(p) == 0.0f)
				continue;
			if (d1->at(p) == 0.0f)
			{
				d1->at(p) = d2->at(p);
				continue;
			}
			d1->at(p) = (d1->at(p) + d2->at(p)) * 0.5f;
		}
	}

	if (conf.debug_lvl > 0)
		std::cout << "SGM took: " << sgm_timer.get_elapsed_sec()
		<< "sec" << std::endl;

	main_view->write_depth_to_view(d1, "smvs-sgm");
}

int smvsrecon(AppSettings& conf)
{
	log_message(conf, "Shading-aware Multi-view Stereo starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("Shading-aware Multi-view Stereo");

	/* Start processing */

	/* Load scene */
	mve::Scene::Ptr scene = mve::Scene::create(conf.path_scene);
	mve::Scene::ViewList& views(scene->get_views());

	/* Check bundle file */
	mve::Bundle::ConstPtr bundle;
	try
	{
		bundle = scene->get_bundle();
	}
	catch (std::exception e)
	{
		bundle = nullptr;
// 		std::cout << "Cannot load bundle file, forcing SGM." << std::endl;
		log_message(conf, "Cannot load bundle file, forcing SGM.");

		conf.use_sgm = true;
		if (conf.sgm_max == 0.0)
		{
			std::cout << "Error: No bundle file and SGM depth given, "
				"please use the --sgm-range option to specify the "
				"depth sweep range for SGM." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	log_message(conf, "smvs-input scale:" + util::string::get(conf.input_scale) + "smvs-output scale:" + util::string::get(conf.output_scale));
	/* Reconstruct all views if no specific list is given */
	if (conf.view_ids.empty()) {
		for (auto view : views) {
			if (view != nullptr && view->is_camera_valid()) {
				conf.view_ids.push_back(view->get_id());
			}
		}
	}

	/* Update legacy data */
	for (std::size_t i = 0; i < views.size(); ++i)
	{
		mve::View::Ptr view = views[i];
		if (view == nullptr) {
			continue;
		}

		mve::View::ImageProxies proxies = view->get_images();
		for (std::size_t p = 0; p < proxies.size(); ++p)
		{
			if (proxies[p].name.compare("lighting-shaded") == 0
				|| proxies[p].name.compare("lighting-sphere") == 0
				|| proxies[p].name.compare("implicit-albedo") == 0)
			{
				view->remove_image(proxies[p].name);
			}
		}

		view->save_view();
		for (std::size_t p = 0; p < proxies.size(); ++p)
		{
			if (proxies[p].name.compare("sgm-depth") == 0)
			{
				std::string file = util::fs::join_path(
					view->get_directory(), proxies[p].filename);
				std::string new_file = util::fs::join_path(
					view->get_directory(), "smvs-sgm.mvei");
				util::fs::rename(file.c_str(), new_file.c_str());
			}
		}
		view->reload_view();
	}

	if (conf.clean_scene)
	{
// 		std::cout << "Cleaning Scene, removing all result embeddings." << std::endl;
		log_message(conf, "Cleaning Scene, removing all result embeddings.");

		for (std::size_t i = 0; i < views.size(); ++i)
		{
			mve::View::Ptr view = views[i];
			if (view == nullptr) {
				continue;
			}
			mve::View::ImageProxies proxies = view->get_images();
			for (std::size_t p = 0; p < proxies.size(); ++p)
			{
				std::string name = proxies[p].name;
				std::string left = util::string::left(name, 4);
				if (left.compare("smvs") == 0)
					view->remove_image(name);
			}
			view->save_view();
		}
		std::exit(EXIT_SUCCESS);
	}

	/* Scale input images */
	if (conf.input_scale < 0)
	{
		double avg_image_size = 0;
		int view_counter = 0;
		for (std::size_t i = 0; i < views.size(); ++i)
		{
			mve::View::Ptr view = views[i];
			if (view == nullptr || !view->has_image(conf.image_embedding)) {
				continue;
			}

			mve::View::ImageProxy const* proxy = view->get_image_proxy(conf.image_embedding);
			uint32_t size = proxy->width * proxy->height;
			avg_image_size += static_cast<double>(size);
			view_counter += 1;
		}
		avg_image_size /= static_cast<double>(view_counter);

		if (avg_image_size > conf.max_pixels) {
			conf.input_scale = std::ceil(std::log2(avg_image_size / conf.max_pixels) / 2);
		} 
		else {
			conf.input_scale = 0;
		}
// 		std::cout << "Automatic input scale: " << conf.input_scale << std::endl;
		log_message(conf, "Automatic input scale: " + util::string::get(conf.input_scale));
	}

	std::string input_name;
	if (conf.input_scale > 0)
		input_name = "undist-L" + util::string::get(conf.input_scale);
	else
		input_name = conf.image_embedding;
	std::cout << "Input embedding: " << input_name << std::endl;

	std::string output_name;
	if (conf.use_shading)
		output_name = "smvs-S" + util::string::get(conf.input_scale);
	else
		output_name = "smvs-B" + util::string::get(conf.input_scale);
	std::cout << "Output embedding: " << output_name << std::endl;

	/* Clean view id list */
	std::vector<int> ignore_list;
	for (std::size_t v = 0; v < conf.view_ids.size(); ++v)
	{
		int const i = conf.view_ids[v];
		if (i > static_cast<int>(views.size() - 1) || views[i] == nullptr)
		{
			std::cout << "View ID " << i << " invalid, skipping view." << std::endl;
			ignore_list.push_back(i);
			continue;
		}
		if (!views[i]->has_image(conf.image_embedding))
		{
			std::cout << "View ID " << i << " missing image embedding, " << "skipping view." << std::endl;
			ignore_list.push_back(i);
			continue;
		}
	}

	for (auto const& id : ignore_list)
		conf.view_ids.erase(std::remove(conf.view_ids.begin(),
			conf.view_ids.end(), id));

	/* Add views to reconstruction list */
	std::vector<int> reconstruction_list;
	int already_reconstructed = 0;
	for (std::size_t v = 0; v < conf.view_ids.size(); ++v)
	{
		int const i = conf.view_ids[v];
		/* Only reconstruct missing views or if forced */
		if (conf.force_recon || !views[i]->has_image(output_name))
			reconstruction_list.push_back(i);
		else
			already_reconstructed += 1;
	}
	if (already_reconstructed > 0)
		std::cout << "Skipping " << already_reconstructed << " views that are already reconstructed." << std::endl;

	/* Create reconstruction threads */
	ThreadPool thread_pool(std::max<std::size_t>(conf.num_threads, 1));

	/* View selection */
	smvs::ViewSelection::Options view_select_opts;
	view_select_opts.num_neighbors = conf.num_neighbors;
	view_select_opts.embedding = conf.image_embedding;
	smvs::ViewSelection view_selection(view_select_opts, views, bundle);
	std::vector<mve::Scene::ViewList> view_neighbors(reconstruction_list.size());
	std::vector<std::future<void>> selection_tasks;
	for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
	{
		int const i = reconstruction_list[v];
		selection_tasks.emplace_back(thread_pool.add_task(
			[i, v, &views, &view_selection, &view_neighbors]
		{
			view_neighbors[v] = view_selection.get_neighbors_for_view(i);
		}));
	}

	if (selection_tasks.size() > 0)
	{
		std::cout << "Running view selection for " << selection_tasks.size() << " views... " << std::flush;
		util::WallTimer timer;
		for (auto && task : selection_tasks) task.get();
		std::cout << " done, took " << timer.get_elapsed_sec() << "s." << std::endl;
	}
	std::vector<int> skipped;
	std::vector<int> final_reconstruction_list;
	std::vector<mve::Scene::ViewList> final_view_neighbors;
	for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
		if (view_neighbors[v].size() < conf.min_neighbors)
			skipped.push_back(reconstruction_list[v]);
		else
		{
			final_reconstruction_list.push_back(reconstruction_list[v]);
			final_view_neighbors.push_back(view_neighbors[v]);
		}
	if (skipped.size() > 0)
	{
		std::cout << "Skipping " << skipped.size() << " views with " << "insufficient number of neighbors." << std::endl;
		std::cout << "Skipped IDs: ";
		for (std::size_t s = 0; s < skipped.size(); ++s)
		{
			std::cout << skipped[s] << " ";
			if (s > 0 && s % 12 == 0)
				std::cout << std::endl << "     ";
		}
		std::cout << std::endl;
	}
	reconstruction_list = final_reconstruction_list;
	view_neighbors = final_view_neighbors;

	/* Create input embedding and resize */
	std::set<int> check_embedding_list;
	for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
	{
		check_embedding_list.insert(reconstruction_list[v]);
		for (auto & neighbor : view_neighbors[v])
			check_embedding_list.insert(neighbor->get_id());
	}
	std::vector<std::future<void>> resize_tasks;
	for (auto const& i : check_embedding_list)
	{
		mve::View::Ptr view = views[i];
		if (view == nullptr
			|| !view->has_image(conf.image_embedding)
			|| view->has_image(input_name))
			continue;

		resize_tasks.emplace_back(thread_pool.add_task(
			[view, &input_name, &conf]
		{
			mve::ByteImage::ConstPtr input =
				view->get_byte_image(conf.image_embedding);
			mve::ByteImage::Ptr scld = input->duplicate();
			for (int i = 0; i < conf.input_scale; ++i)
				scld = mve::image::rescale_half_size_gaussian<uint8_t>(scld);
			view->set_image(scld, input_name);
			view->save_view();
		}));
	}
	if (resize_tasks.size() > 0)
	{
		std::cout << "Resizing input images for " << resize_tasks.size() << " views... " << std::flush;
		util::WallTimer timer;
		for (auto && task : resize_tasks) task.get();
		std::cout << " done, took " << timer.get_elapsed_sec() << "s." << std::endl;
	}

	std::vector<std::future<void>> results;
	std::mutex counter_mutex;
	std::size_t started = 0;
	std::size_t finished = 0;
	util::WallTimer timer;

	for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
	{
		int const i = reconstruction_list[v];

		results.emplace_back(thread_pool.add_task(
			[v, i, &views, &conf, &counter_mutex, &input_name, &output_name,
			&started, &finished, &reconstruction_list, &view_neighbors,
			bundle, scene]
		{
			smvs::StereoView::Ptr main_view = smvs::StereoView::create(
				views[i], input_name, conf.use_shading,
				conf.gamma_correction);
			mve::Scene::ViewList neighbors = view_neighbors[v];

			std::vector<smvs::StereoView::Ptr> stereo_views;

			std::unique_lock<std::mutex> lock(counter_mutex);
			std::cout << "\rStarting "
				<< ++started << "/" << reconstruction_list.size()
				<< " ID: " << i
				<< " Neighbors: ";
			for (std::size_t n = 0; n < conf.num_neighbors
				&& n < neighbors.size(); ++n)
				std::cout << neighbors[n]->get_id() << " ";
			std::cout << std::endl;
			lock.unlock();

			for (std::size_t n = 0; n < conf.num_neighbors
				&& n < neighbors.size(); ++n)
			{
				smvs::StereoView::Ptr sv = smvs::StereoView::create(
					neighbors[n], input_name);
				stereo_views.push_back(sv);
			}

			if (conf.use_sgm)
			{
				int sgm_width = views[i]->get_image_proxy(input_name)->width;
				int sgm_height = views[i]->get_image_proxy(input_name)->height;
				for (int scale = 0; scale < conf.sgm_scale; ++scale)
				{
					sgm_width = (sgm_width + 1) / 2;
					sgm_height = (sgm_height + 1) / 2;
				}
				if (conf.force_sgm || !views[i]->has_image("smvs-sgm")
					|| views[i]->get_image_proxy("smvs-sgm")->width !=
					sgm_width
					|| views[i]->get_image_proxy("smvs-sgm")->height !=
					sgm_height)
					reconstruct_sgm_depth_for_view(conf, main_view, stereo_views, bundle);
			}

			smvs::DepthOptimizer::Options do_opts;
			do_opts.regularization = 0.01 * conf.regularization;
			do_opts.num_iterations = 5;
			do_opts.debug_lvl = conf.debug_lvl;
			do_opts.min_scale = conf.output_scale;
			do_opts.use_shading = conf.use_shading;
			do_opts.output_name = output_name;
			do_opts.use_sgm = conf.use_sgm;
			do_opts.full_optimization = conf.full_optimization;
			do_opts.light_surf_regularization = conf.light_surf_regularization;

			smvs::DepthOptimizer optimizer(main_view, stereo_views,
				bundle, do_opts);
			optimizer.optimize();

			std::unique_lock<std::mutex> lock2(counter_mutex);
			std::cout << "\rFinished "
				<< ++finished << "/" << reconstruction_list.size()
				<< " ID: " << i
				<< std::endl;
			lock2.unlock();
		}));
	}
	/* Wait for reconstruction to finish */
	for (auto && result : results) result.get();

	/* Save results */
	if (results.size() > 0)
	{
		scene->save_views();
		std::cout << "Done. Reconstruction took: " << timer.get_elapsed_sec() << "s" << std::endl;
	}
	else {
		std::cout << "All valid views are already reconstructed." << std::endl;
	}

	/* Generate mesh */
	if (!conf.recon_only) {
		generate_mesh(conf, scene, input_name, output_name);
	}

	log_message(conf, "Shading-aware Multi-view Stereo ends.");
}

int dmrecon(AppSettings& conf)
{
	log_message(conf, "Depthmap reconstruction starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("MVE Depth Map Reconstruction");


	conf.mvs.useColorScale = true;// false;
	conf.max_pixels = 0;
	conf.mvs.scale = 2;//3
	conf.mvs.keepDzMap = true;
	conf.mvs.keepConfidenceMap = true;
	conf.write_ply = false;// true;
	conf.progress_style = PROGRESS_SIMPLE; //PROGRESS_SILENT;
	conf.force_recon = false;// true;

	/* don't show progress twice */
	if (conf.progress_style != PROGRESS_SIMPLE) {
		conf.mvs.quiet = true;
	}

	
	/* Load MVE scene. */
	mve::Scene::Ptr scene;
	try {
		scene = mve::Scene::create(conf.path_scene);
		scene->get_bundle();
	}
	catch (std::exception& e) {
// 		std::cerr << "Error loading scene: " << e.what() << std::endl;
		log_message(conf, "Error loading scene: " + util::string::get(e.what()));

		return EXIT_FAILURE;
	}

	/* Settings for Multi-view stereo */
	conf.mvs.writePlyFile = conf.write_ply;
	conf.mvs.plyPath = util::fs::join_path(conf.path_scene, conf.ply_dest);

	fancyProgressPrinter.setBasePath(conf.path_scene);
	fancyProgressPrinter.setNumViews(scene->get_views().size());
	if (conf.progress_style == PROGRESS_FANCY) {
		fancyProgressPrinter.start();
	}

	util::WallTimer timer;
	if (conf.master_id >= 0) {
		/* Calculate scale from max pixels. */
		if (conf.max_pixels > 0) {
			conf.mvs.scale = get_scale_from_max_pixels(scene, conf, conf.mvs);
		}

// 		std::cout << "Reconstructing view ID " << conf.master_id << std::endl;
		log_message(conf, "Reconstructing view ID " + util::string::get(conf.master_id));

		conf.mvs.refViewNr = (std::size_t)conf.master_id;
		fancyProgressPrinter.addRefView(conf.master_id);
		try {
			reconstruct(scene, conf.mvs);
		}
		catch (std::exception &err) {
// 			std::cerr << err.what() << std::endl;
			log_message(conf, "Error reconstruct: " + util::string::get(err.what()));
			return EXIT_FAILURE;
		}
	}
	else {
		mve::Scene::ViewList& views(scene->get_views());
		if (conf.view_ids.empty()) {
// 			std::cout << "Reconstructing all views..." << std::endl;
			log_message(conf, "Reconstructing all views...");
			for (std::size_t i = 0; i < views.size(); ++i) {
				conf.view_ids.push_back(i);
			}
		}
		else {
// 			std::cout << "Reconstructing views from list..." << std::endl;
			log_message(conf, "Reconstructing views from list...");
		}
		fancyProgressPrinter.addRefViews(conf.view_ids);

#pragma omp parallel for schedule(dynamic, 1)
#if !defined(_MSC_VER)
		for (std::size_t i = 0; i < conf.view_ids.size(); ++i)
#else
		for (int64_t i = 0; i < conf.view_ids.size(); ++i)
#endif
		{
			std::size_t id = conf.view_ids[i];
			if (id >= views.size())
			{
// 				std::cout << "Invalid ID " << id << ", skipping!" << std::endl;
				log_message(conf, "Invalid ID " + util::string::get(id) + ", skipping!");
				continue;
			}

			if (views[id] == nullptr || !views[id]->is_camera_valid()) {
				continue;
			}

			/* Setup MVS. */
			mvs::Settings settings(conf.mvs);
			settings.refViewNr = id;
			if (conf.max_pixels > 0) {
				settings.scale = get_scale_from_max_pixels(scene, conf, settings);
			}

			std::string embedding_name = "depth-L" + util::string::get(settings.scale);
			if (!conf.force_recon && views[id]->has_image(embedding_name)) {
				continue;
			}

			try
			{
				reconstruct(scene, settings);
				views[id]->save_view();
			}
			catch (std::exception &err)
			{
				std::cerr << err.what() << std::endl;
				log_message(conf, util::string::get(err.what()));
			}
		}
	}

	if (conf.progress_style == PROGRESS_FANCY) {
		fancyProgressPrinter.stop();
	}

// 	std::cout << "Reconstruction took " << timer.get_elapsed() << "ms." << std::endl;
	log_message(conf, "Reconstruction took " + util::string::get(timer.get_elapsed()) + "ms.");

	/* Save scene */
// 	std::cout << "Saving views back to disc..." << std::endl;
	log_message(conf, "Saving views back to disc...");

	scene->save_views();

	log_message(conf, "Depthmap reconstruction is done.");

	return EXIT_SUCCESS;
}

int scene2pset(AppSettings& conf)
{
	log_message(conf, "Scene to point set starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("MVE Scene to Pointset");

	/* Init default settings. */
	conf.pset_name = util::fs::join_path(conf.path_scene, "pset-L2.ply");
	conf.with_conf = true;
	conf.with_normals = true;
	conf.with_scale = true;
	int const scale = conf.mvs.scale;//2;
	conf.dmname = "depth-L" + util::string::get<int>(scale);
	conf.image = (scale == 0) ? "undistorted" : "undist-L" + util::string::get<int>(scale);

	if (util::string::right(conf.pset_name, 5) == ".npts" || util::string::right(conf.pset_name, 6) == ".bnpts")
	{
		conf.with_normals = true;
		conf.with_scale = false;
		conf.with_conf = false;
	}

	if (conf.poisson_normals)
	{
		conf.with_normals = true;
		conf.with_conf = true;
	}

	/* If requested, use given AABB. */
	math::Vec3f aabbmin, aabbmax;
	if (!conf.aabb.empty()) {
		aabb_from_string(conf.aabb, &aabbmin, &aabbmax);
	}

// 	std::cout << "Using depthmap \"" << conf.dmname << "\" and color image \"" << conf.image << "\"" << std::endl;
	log_message(conf, "Using depthmap \"" + util::string::get(conf.dmname) + "\" and color image \"" + util::string::get(conf.image) + "\"");

	/* Prepare output mesh. */
	mve::TriangleMesh::Ptr pset(mve::TriangleMesh::create());
	mve::TriangleMesh::VertexList& verts(pset->get_vertices());
	mve::TriangleMesh::NormalList& vnorm(pset->get_vertex_normals());
	mve::TriangleMesh::ColorList& vcolor(pset->get_vertex_colors());
	mve::TriangleMesh::ValueList& vvalues(pset->get_vertex_values());
	mve::TriangleMesh::ConfidenceList& vconfs(pset->get_vertex_confidences());

	/* Load scene. */
	mve::Scene::Ptr scene = mve::Scene::create(conf.path_scene);

	/* Iterate over views and get points. */
	mve::Scene::ViewList& views(scene->get_views());
#pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
	for (std::size_t i = 0; i < views.size(); ++i)
#else
	for (int64_t i = 0; i < views.size(); ++i)
#endif
	{
		mve::View::Ptr view = views[i];
		if (view == nullptr) {
			continue;
		}

		std::size_t view_id = view->get_id();
		if (!conf.ids.empty() && std::find(conf.ids.begin(), conf.ids.end(), view_id) == conf.ids.end()) {
			continue;
		}

		mve::CameraInfo const& cam = view->get_camera();
		if (cam.flen == 0.0f) {
			continue;
		}

		mve::FloatImage::Ptr dm = view->get_float_image(conf.dmname);
		if (dm == nullptr) {
			continue;
		}

		if (conf.min_valid_fraction > 0.0f)
		{
			float num_total = static_cast<float>(dm->get_value_amount());
			float num_recon = 0;
			for (int j = 0; j < dm->get_value_amount(); ++j) {
				if (dm->at(j) > 0.0f) {
					num_recon += 1.0f;
				}
			}

			float fraction = num_recon / num_total;
			if (fraction < conf.min_valid_fraction)
			{
// 				std::cout << "View " << view->get_name() << ": Fill status " << util::string::get_fixed(fraction * 100.0f, 2) << "%, skipping." << std::endl;
				log_message(conf, "View " + util::string::get(view->get_name()) + ": Fill status " + util::string::get(util::string::get_fixed(fraction * 100.0f, 2)) + "%, skipping.");
				continue;
			}
		}

		mve::ByteImage::Ptr ci;
		if (!conf.image.empty()) {
			ci = view->get_byte_image(conf.image);
		}

#pragma omp critical
		std::cout << "Processing view \"" << view->get_name() << "\"" << (ci != nullptr ? " (with colors)" : "") << "..." << std::endl;

		/* Triangulate depth map. */
		mve::TriangleMesh::Ptr mesh;
		mesh = mve::geom::depthmap_triangulate(dm, ci, cam);
		mve::TriangleMesh::VertexList const& mverts(mesh->get_vertices());
		mve::TriangleMesh::NormalList const& mnorms(mesh->get_vertex_normals());
		mve::TriangleMesh::ColorList const& mvcol(mesh->get_vertex_colors());
		mve::TriangleMesh::ConfidenceList& mconfs(mesh->get_vertex_confidences());

		if (conf.with_normals) {
			mesh->ensure_normals();
		}

		/* If confidence is requested, compute it. */
		if (conf.with_conf)
		{
			/* Per-vertex confidence down-weighting boundaries. */
			mve::geom::depthmap_mesh_confidences(mesh, 4);

#if 0
			/* Per-vertex confidence based on normal-viewdir dot product. */
			mesh->ensure_normals();
			math::Vec3f campos;
			cam.fill_camera_pos(*campos);
			for (std::size_t i = 0; i < mverts.size(); ++i)
				mconfs[i] *= (campos - mverts[i]).normalized().dot(mnorms[i]);
#endif
		}

		if (conf.poisson_normals) {
			poisson_scale_normals(mconfs, &mesh->get_vertex_normals());
		}

		/* If scale is requested, compute it. */
		std::vector<float> mvscale;
		if (conf.with_scale)
		{
			mvscale.resize(mverts.size(), 0.0f);
			mve::MeshInfo mesh_info(mesh);
			for (std::size_t j = 0; j < mesh_info.size(); ++j)
			{
				mve::MeshInfo::VertexInfo const& vinf = mesh_info[j];
				for (std::size_t k = 0; k < vinf.verts.size(); ++k) {
					mvscale[j] += (mverts[j] - mverts[vinf.verts[k]]).norm();
				}
				mvscale[j] /= static_cast<float>(vinf.verts.size());
				mvscale[j] *= conf.scale_factor;
			}
		}

		/* Add vertices and optional colors and normals to mesh. */
		if (conf.aabb.empty())
		{
			/* Fast if no bounding box is given. */
#pragma omp critical
		{
			verts.insert(verts.end(), mverts.begin(), mverts.end());
			if (!mvcol.empty())
				vcolor.insert(vcolor.end(), mvcol.begin(), mvcol.end());
			if (conf.with_normals)
				vnorm.insert(vnorm.end(), mnorms.begin(), mnorms.end());
			if (conf.with_scale)
				vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());
			if (conf.with_conf)
				vconfs.insert(vconfs.end(), mconfs.begin(), mconfs.end());
		}
		}
		else
		{
			/* Check every point if a bounding box is given.  */
			for (std::size_t i = 0; i < mverts.size(); ++i)
			{
				if (!math::geom::point_box_overlap(mverts[i], aabbmin, aabbmax))
					continue;

#pragma omp critical
				{
					verts.push_back(mverts[i]);
					if (!mvcol.empty())
						vcolor.push_back(mvcol[i]);
					if (conf.with_normals)
						vnorm.push_back(mnorms[i]);
					if (conf.with_scale)
						vvalues.push_back(mvscale[i]);
					if (conf.with_conf)
						vconfs.push_back(mconfs[i]);
				}
			}
		}

		dm.reset();
		ci.reset();
		view->cache_cleanup();
	}

	/* If a mask is given, clip vertices with the masks in all images. */
	if (!conf.mask.empty())
	{
		std::cout << "Filtering points using silhouette masks..." << std::endl;
		std::vector<bool> delete_list(verts.size(), false);
		std::size_t num_filtered = 0;

		for (std::size_t i = 0; i < views.size(); ++i)
		{
			mve::View::Ptr view = views[i];
			if (view == nullptr || view->get_camera().flen == 0.0f) {
				continue;
			}

			mve::ByteImage::Ptr mask = view->get_byte_image(conf.mask);
			if (mask == nullptr)
			{
				std::cout << "Mask not found for image \"" << view->get_name() << "\", skipping." << std::endl;
				continue;
			}
			if (mask->channels() != 1)
			{
				std::cout << "Expected 1-channel mask for image \"" << view->get_name() << "\", skipping." << std::endl;
				continue;
			}

// 			std::cout << "Processing mask for \"" << view->get_name() << "\"..." << std::endl;
			log_message(conf, "Processing mask for \"" + util::string::get(view->get_name()) + "\"...");

			mve::CameraInfo cam = view->get_camera();
			math::Matrix4f wtc;
			cam.fill_world_to_cam(*wtc);
			math::Matrix3f calib;
			cam.fill_calibration(*calib, mask->width(), mask->height());

			/* Iterate every point and check with mask. */
			for (std::size_t j = 0; j < verts.size(); ++j)
			{
				if (delete_list[j]) {
					continue;
				}

				math::Vec3f p = calib * wtc.mult(verts[j], 1.0f);
				p[0] = p[0] / p[2];
				p[1] = p[1] / p[2];

				if (p[0] < 0.0f || p[1] < 0.0f || p[0] >= mask->width() || p[1] >= mask->height()) {
					continue;
				}

				int const ix = static_cast<int>(p[0]);
				int const iy = static_cast<int>(p[1]);
				if (mask->at(ix, iy, 0) == 0)
				{
					delete_list[j] = true;
					num_filtered += 1;
				}
			}
			view->cache_cleanup();
		}
		pset->delete_vertices(delete_list);
// 		std::cout << "Filtered a total of " << num_filtered << " points." << std::endl;
		log_message(conf, "Filtered a total of " + util::string::get(num_filtered) + " points.");
	}

	/* Write mesh to disc. */
// 	std::cout << "Writing final point set (" << verts.size() << " points)..." << std::endl;
	log_message(conf, "Writing final point set (" + util::string::get(verts.size()) + " points)...");

	if (util::string::right(conf.pset_name, 4) == ".ply")
	{
		mve::geom::SavePLYOptions opts;
		opts.write_vertex_normals = conf.with_normals;
		opts.write_vertex_values = conf.with_scale;
		opts.write_vertex_confidences = conf.with_conf;
		mve::geom::save_ply_mesh(pset, conf.pset_name, opts);
	}
	else
	{
		mve::geom::save_mesh(pset, conf.pset_name);
	}

	log_message(conf, "Scene to point set ends.");

	return EXIT_SUCCESS;
}


int fssrecon(AppSettings& conf/*, fssr::SampleIO::Options const& pset_opts*/)
{
	log_message(conf, "Floating Scale Surface Reconstruction starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("Floating Scale Surface Reconstruction");

	/* Init default settings. */
	fssr::SampleIO::Options pset_opts;
	std::string in_mesh = conf.pset_name;//util::fs::join_path(app_opts.path_scene, "pset-L2.ply");////
	std::string out_mesh = util::fs::join_path(conf.path_scene, "surface-L2.ply");
	conf.in_files.push_back(in_mesh);
	conf.in_files.push_back(out_mesh);
	if (conf.in_files.size() < 2)
	{
		return EXIT_FAILURE;
	}
	conf.out_mesh = conf.in_files.back();
	conf.in_files.pop_back();

	if (conf.refine_octree < 0 || conf.refine_octree > 3)
	{
// 		std::cerr << "Unreasonable refine level of " << conf.refine_octree << ", exiting." << std::endl;
		log_message(conf, "Unreasonable refine level of " + util::string::get(conf.refine_octree) + ", exiting.");
		return EXIT_FAILURE;
	}

	/* Load input point set and insert samples in the octree. */
	fssr::IsoOctree octree;
	for (std::size_t i = 0; i < conf.in_files.size(); ++i)
	{
// 		std::cout << "Loading: " << conf.in_files[i] << "..." << std::endl;
		log_message(conf, "Loading: " + util::string::get(conf.in_files[i]) + "...");
		util::WallTimer timer;

		fssr::SampleIO loader(pset_opts);
		loader.open_file(conf.in_files[i]);
		fssr::Sample sample;
		while (loader.next_sample(&sample)) {
			octree.insert_sample(sample);
		}

// 		std::cout << "Loading samples took " << timer.get_elapsed() << "ms." << std::endl;
		log_message(conf, "Loading samples took " + util::string::get(timer.get_elapsed()) + "ms.");
	}

	/* Exit if no samples have been inserted. */
	if (octree.get_num_samples() == 0)
	{
// 		std::cerr << "Octree does not contain any samples, exiting." << std::endl;
		log_message(conf, "Octree does not contain any samples, exiting.");
		std::exit(EXIT_FAILURE);
	}

	/* Refine octree if requested. Each iteration adds one level. */
	if (conf.refine_octree > 0)
	{
		std::cout << "Refining octree..." << std::flush;
		util::WallTimer timer;
		for (int i = 0; i < conf.refine_octree; ++i) {
			octree.refine_octree();
		}
		std::cout << " took " << timer.get_elapsed() << "ms" << std::endl;
	}

	/* Compute voxels. */
	octree.limit_octree_level();
	octree.print_stats(std::cout);
	octree.compute_voxels();
	octree.clear_samples();

	/* Extract isosurface. */
	mve::TriangleMesh::Ptr mesh;
	{
// 		std::cout << "Extracting isosurface..." << std::endl;
		log_message(conf, "Extracting isosurface...");

		util::WallTimer timer;
		fssr::IsoSurface iso_surface(&octree, conf.interp_type);
		mesh = iso_surface.extract_mesh();

// 		std::cout << "  Done. Surface extraction took " << timer.get_elapsed() << "ms." << std::endl;
		log_message(conf, "Done. Surface extraction took " + util::string::get(timer.get_elapsed()) + "ms.");
	}
	octree.clear();

	/* Check if anything has been extracted. */
	if (mesh->get_vertices().empty())
	{
// 		std::cerr << "Isosurface does not contain any vertices, exiting." << std::endl;
		log_message(conf, "Isosurface does not contain any vertices, exiting.");
		std::exit(EXIT_FAILURE);
	}

	/* Surfaces between voxels with zero confidence are ghosts. */
	{
		std::cout << "Deleting zero confidence vertices..." << std::flush;
		util::WallTimer timer;
		std::size_t num_vertices = mesh->get_vertices().size();
		mve::TriangleMesh::DeleteList delete_verts(num_vertices, false);
		for (std::size_t i = 0; i < num_vertices; ++i) {
			if (mesh->get_vertex_confidences()[i] == 0.0f) {
				delete_verts[i] = true;
			}
		}

		mesh->delete_vertices_fix_faces(delete_verts);
		std::cout << " took " << timer.get_elapsed() << "ms." << std::endl;
	}

	/* Check for color and delete if not existing. */
	mve::TriangleMesh::ColorList& colors = mesh->get_vertex_colors();
	if (!colors.empty() && colors[0].minimum() < 0.0f)
	{
		std::cout << "Removing dummy mesh coloring..." << std::endl;
		colors.clear();
	}

	/* Write output mesh. */
	mve::geom::SavePLYOptions ply_opts;
	ply_opts.write_vertex_colors = true;
	ply_opts.write_vertex_confidences = true;
	ply_opts.write_vertex_values = true;
	mve::geom::save_ply_mesh(mesh, conf.out_mesh, ply_opts);

// 	std::cout << "All done. Remember to clean the output mesh." << std::endl;
	log_message(conf, "All done. Remember to clean the output mesh.");
	log_message(conf, "Floating Scale Surface Reconstruction ends.");

	return EXIT_SUCCESS;
}

int mesh_clean(AppSettings& conf)
{
	log_message(conf, "FSSR Mesh Cleaning starts.");

// 	util::system::register_segfault_handler();
// 	util::system::print_build_timestamp("MVE FSSR Mesh Cleaning");

// 	conf.out_mesh_clean = ;

	conf.conf_threshold = 10;
	conf.clean_degenerated = false;
	conf.delete_scale = true;
	conf.delete_conf = true;
	conf.delete_colors = false;//true
	std::string cleaned_mesh_file = util::fs::join_path(conf.path_scene, "clean.ply");
	/* Load input mesh. */
	mve::TriangleMesh::Ptr mesh;
	try
	{
		std::cout << "Loading mesh: " << conf.out_mesh << std::endl;
		mesh = mve::geom::load_mesh(conf.out_mesh);
	}
	catch (std::exception& e)
	{
		std::cerr << "Error loading mesh: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	/* Sanity checks. */
	if (mesh->get_vertices().empty())
	{
		std::cerr << "Error: Mesh is empty!" << std::endl;
		return EXIT_FAILURE;
	}

	if (!mesh->has_vertex_confidences() && conf.conf_threshold > 0.0f)
	{
		std::cerr << "Error: Confidence cleanup requested, but mesh "
			"has no confidence values." << std::endl;
		return EXIT_FAILURE;
	}

	if (mesh->get_faces().empty()
		&& (conf.clean_degenerated || conf.component_size > 0))
	{
		std::cerr << "Error: Components/faces cleanup "
			"requested, but mesh has no faces." << std::endl;
		return EXIT_FAILURE;
	}

	/* Remove low-confidence geometry. */
	if (conf.conf_percentile > 0.0f)
		conf.conf_threshold = percentile(mesh->get_vertex_confidences(),
			conf.conf_percentile);
	if (conf.conf_threshold > 0.0f)
	{
		std::cout << "Removing low-confidence geometry (threshold "
			<< conf.conf_threshold << ")..." << std::endl;
		std::size_t num_verts = mesh->get_vertices().size();
		remove_low_conf_vertices(mesh, conf.conf_threshold);
		std::size_t new_num_verts = mesh->get_vertices().size();
		std::cout << "  Deleted " << (num_verts - new_num_verts)
			<< " low-confidence vertices." << std::endl;
	}

	/* Remove isolated components if requested. */
	if (conf.component_size > 0)
	{
		std::cout << "Removing isolated components below "
			<< conf.component_size << " vertices..." << std::endl;
		std::size_t num_verts = mesh->get_vertices().size();
		mve::geom::mesh_components(mesh, conf.component_size);
		std::size_t new_num_verts = mesh->get_vertices().size();
		std::cout << "  Deleted " << (num_verts - new_num_verts)
			<< " vertices in isolated regions." << std::endl;
	}

	/* Remove degenerated faces from the mesh. */
	if (conf.clean_degenerated)
	{
		std::cout << "Removing degenerated faces..." << std::endl;
		std::size_t num_collapsed = fssr::clean_mc_mesh(mesh);
		std::cout << "  Collapsed " << num_collapsed << " edges." << std::endl;
	}

	/* Write output mesh. */
	std::cout << "Writing mesh: " << cleaned_mesh_file << std::endl;
	if (util::string::right(cleaned_mesh_file, 4) == ".ply")
	{
		mve::geom::SavePLYOptions ply_opts;
		ply_opts.write_vertex_colors = !conf.delete_colors;
		ply_opts.write_vertex_confidences = !conf.delete_conf;
		ply_opts.write_vertex_values = !conf.delete_scale;
		mve::geom::save_ply_mesh(mesh, cleaned_mesh_file, ply_opts);
	}
	else
	{
		mve::geom::save_mesh(mesh, cleaned_mesh_file);
	}

	log_message(conf, "FSSR Mesh Cleaning ends.");

	return EXIT_SUCCESS;
}

void texRecon(std::string &in_scene, std::string &in_mesh, std::string &out_prefix)
{
	//util::system::print_build_timestamp(argv[0]);
	util::system::register_segfault_handler();

#ifdef RESEARCH
	std::cout << "******************************************************************************" << std::endl
		<< " Due to use of the -DRESEARCH=ON compile option, this program is licensed " << std::endl
		<< " for research purposes only. Please pay special attention to the gco license." << std::endl
		<< "******************************************************************************" << std::endl;
#endif

	Timer timer;
	util::WallTimer wtimer;

	Arguments conf;
	conf.in_scene = in_scene;
	conf.in_mesh = in_mesh;
	conf.out_prefix = out_prefix;
	if (!util::fs::dir_exists(util::fs::dirname(conf.out_prefix).c_str())) {
		std::cerr << "Destination directory does not exist!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::cout << "Load and prepare mesh: " << std::endl;
	mve::TriangleMesh::Ptr mesh;
	try {
		mesh = mve::geom::load_ply_mesh(conf.in_mesh);
	}
	catch (std::exception& e) {
		std::cerr << "\tCould not load mesh: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	mve::MeshInfo mesh_info(mesh);
	tex::prepare_mesh(&mesh_info, mesh);

	std::cout << "Generating texture views: " << std::endl;
	tex::TextureViews texture_views;
	tex::generate_texture_views(conf.in_scene, &texture_views);

	//write_string_to_file(conf.out_prefix + ".conf", conf.to_string());
	timer.measure("Loading");

	std::size_t const num_faces = mesh->get_faces().size() / 3;

	std::cout << "Building adjacency graph: " << std::endl;
	tex::Graph graph(num_faces);
	tex::build_adjacency_graph(mesh, mesh_info, &graph);

	if (conf.labeling_file.empty()) {
		std::cout << "View selection:" << std::endl;
		util::WallTimer rwtimer;

		tex::DataCosts data_costs(num_faces, texture_views.size());
		if (conf.data_cost_file.empty()) {
			tex::calculate_data_costs(mesh, &texture_views, conf.settings, &data_costs);

			if (conf.write_intermediate_results) {
				std::cout << "\tWriting data cost file... " << std::flush;
				tex::DataCosts::save_to_file(data_costs, conf.out_prefix + "_data_costs.spt");
				std::cout << "done." << std::endl;
			}
		}
		else {
			std::cout << "\tLoading data cost file... " << std::flush;
			try {
				tex::DataCosts::load_from_file(conf.data_cost_file, &data_costs);
			}
			catch (util::FileException e) {
				std::cout << "failed!" << std::endl;
				std::cerr << e.what() << std::endl;
				std::exit(EXIT_FAILURE);
			}
			std::cout << "done." << std::endl;
		}
		timer.measure("Calculating data costs");

		tex::view_selection(data_costs, &graph, conf.settings);
		timer.measure("Running MRF optimization");
		std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;

		/* Write labeling to file. */
		if (conf.write_intermediate_results) {
			std::vector<std::size_t> labeling(graph.num_nodes());
			for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
				labeling[i] = graph.get_label(i);
			}
			vector_to_file(conf.out_prefix + "_labeling.vec", labeling);
		}
	}
	else {
		std::cout << "Loading labeling from file... " << std::flush;

		/* Load labeling from file. */
		std::vector<std::size_t> labeling = vector_from_file<std::size_t>(conf.labeling_file);
		if (labeling.size() != graph.num_nodes()) {
			std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		/* Transfer labeling to graph. */
		for (std::size_t i = 0; i < labeling.size(); ++i) {
			const std::size_t label = labeling[i];
			if (label > texture_views.size()) {
				std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			graph.set_label(i, label);
		}

		std::cout << "done." << std::endl;
	}

	tex::TextureAtlases texture_atlases;
	{
		/* Create texture patches and adjust them. */
		tex::TexturePatches texture_patches;
		tex::VertexProjectionInfos vertex_projection_infos;
		std::cout << "Generating texture patches:" << std::endl;
		tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
			conf.settings, &vertex_projection_infos, &texture_patches);

		if (conf.settings.global_seam_leveling) {
			std::cout << "Running global seam leveling:" << std::endl;
			tex::global_seam_leveling(graph, mesh, mesh_info, vertex_projection_infos, &texture_patches);
			timer.measure("Running global seam leveling");
		}
		else {
			ProgressCounter texture_patch_counter("Calculating validity masks for texture patches", texture_patches.size());
/*#pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
			for (std::size_t i = 0; i < texture_patches.size(); ++i) {
#else*/
			for (std::int64_t i = 0; i < texture_patches.size(); ++i) {
//#endif
				texture_patch_counter.progress<SIMPLE>();
				TexturePatch::Ptr texture_patch = texture_patches[i];
				std::vector<math::Vec3f> patch_adjust_values(texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));
				texture_patch->adjust_colors(patch_adjust_values);
				texture_patch_counter.inc();
			}
			timer.measure("Calculating texture patch validity masks");
		}

		if (conf.settings.local_seam_leveling) {
			std::cout << "Running local seam leveling:" << std::endl;
			tex::local_seam_leveling(graph, mesh, vertex_projection_infos, &texture_patches);
		}
		timer.measure("Running local seam leveling");

		/* Generate texture atlases. */
		std::cout << "Generating texture atlases:" << std::endl;
		tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases);
	}

	/* Create and write out obj model. */
	{
		std::cout << "Building objmodel:" << std::endl;
		tex::Model model;
		tex::build_model(mesh, texture_atlases, &model, false);
		timer.measure("Building OBJ model");

		std::cout << "\tSaving model... " << std::flush;
		tex::Model::save(model, conf.out_prefix);
		std::cout << "done." << std::endl;
		timer.measure("Saving");
	}

	std::cout << "Whole texturing procedure took: " << wtimer.get_elapsed_sec() << "s" << std::endl;
	timer.measure("Total");
	/*if (conf.write_timings) {
		timer.write_to_file(conf.out_prefix + "_timings.csv");
	}*/

	if (conf.write_view_selection_model) {
		texture_atlases.clear();
		std::cout << "Generating debug texture patches:" << std::endl;
		{
			tex::TexturePatches texture_patches;
			generate_debug_embeddings(&texture_views);
			tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
			tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
				conf.settings, &vertex_projection_infos, &texture_patches);
			tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases);
		}

		std::cout << "Building debug objmodel:" << std::endl;
		{
			tex::Model model;
			tex::build_model(mesh, texture_atlases, &model, false);
			std::cout << "\tSaving model... " << std::flush;
			tex::Model::save(model, conf.out_prefix + "_view_selection");
			std::cout << "done." << std::endl;
		}
	}
}


int main(int argc, char** argv)
{
	std::string path_image;
	std::string path_scene;

	std::cout << "Input image directory: ";
	std::getline(std::cin, path_image);

	std::cout << "Input scene directory: ";
	std::getline(std::cin, path_scene);
	
	AppSettings conf;
	conf.path_image = util::fs::sanitize_path(path_image);
	conf.path_scene = util::fs::sanitize_path(path_scene);

 	make_scene(conf);
	sfmrecon(conf);
 	//dmrecon(conf);//
 	//scene2pset(conf);//
	smvsrecon(conf);
	fssrecon(conf);
	mesh_clean(conf);
	std::string in_scene = conf.path_scene + "::original";//"f:/dataset/foot/scene_mve::original";
	std::string out_prefix = conf.path_scene + "/prefix";//"f:/dataset/foot/scene_mve/prefix";
	std::string in_mesh = conf.path_scene + "/clean.ply";//"f:/dataset/foot/scene_mve/clean.ply";
	texRecon(in_scene, in_mesh, out_prefix);
	return EXIT_SUCCESS;
}
