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

/*#include "util/system.h"
#include "util/timer.h"
#include "util/string.h"
#include "util/arguments.h"
#include "util/file_system.h"
#include "util/tokenizer.h"

#include "math/algo.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "math/octree_tools.h"
/*
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
#include "mve/mesh_tools.h"*/
#include "mve/scene.h"
#include "mve/view.h"

#include "sfm/nearest_neighbor.h"
#include "sfm/feature_set.h"
#include "sfm/bundler_common.h"
#include "sfm/bundler_features.h"
#include "sfm/bundler_matching.h"
/*#include "sfm/bundler_tracks.h"
#include "sfm/bundler_init_pair.h"
#include "sfm/bundler_intrinsics.h"
#include "sfm/bundler_incremental.h"*/
#include "sfm/ransac.h"
#include "sfm/correspondence.h"

#include "fancy_progress_printer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

struct AppSettings;


void log_message(AppSettings const& conf, std::string const& message);

void cv_features_and_matching(mve::Scene::Ptr scene, AppSettings const& conf, sfm::bundler::Matching &matcher, sfm::bundler::ViewportList* viewports, sfm::bundler::PairwiseMatching* pairwise_matching);
void cv_multi_feature_matching(mve::Scene::Ptr scene, AppSettings const& conf, std::string &img_extension, sfm::bundler::Matching &matcher,
							   sfm::bundler::ViewportList* final_viewports,
							   sfm::bundler::PairwiseMatching* final_pairwise_matching);