#include "common.h"

void getCamPositions(mve::Scene::ViewList& views)
{
	for (size_t i = 0; i < views.size(); ++i)
	{
		float pos[3];
		views.at(i).get()->get_camera().fill_camera_translation(pos);
	}
}

bool filteringMatches(AppSettings const& conf, sfm::bundler::ViewportList* viewports, int viewId1, int viewId2, vector<DMatch> &matches_in,
	sfm::bundler::Matching &matcher, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	/* Build correspondences from feature matching result. */
	sfm::Correspondences2D2D unfiltered_matches;
	sfm::CorrespondenceIndices unfiltered_indices, filtered_indices;
	unfiltered_matches.clear();
	unfiltered_indices.clear();
	filtered_indices.clear();
	for (int i = 0; i < matches_in.size(); ++i)
	{
		sfm::Correspondence2D2D match;
		match.p1[0] = viewports->at(viewId1).features.positions[matches_in[i].queryIdx][0];
		match.p1[1] = viewports->at(viewId1).features.positions[matches_in[i].queryIdx][1];
		match.p2[0] = viewports->at(viewId2).features.positions[matches_in[i].trainIdx][0];
		match.p2[1] = viewports->at(viewId2).features.positions[matches_in[i].trainIdx][1];
		unfiltered_matches.push_back(match);
		unfiltered_indices.push_back(std::make_pair(matches_in[i].queryIdx, matches_in[i].trainIdx));
	}
	if (!matcher.filteringMatches(unfiltered_matches, unfiltered_indices, filtered_indices))
		return false;
	sfm::bundler::TwoViewMatching twoViewMatches;
	twoViewMatches.view_1_id = viewId1;
	twoViewMatches.view_2_id = viewId2;
	twoViewMatches.matches.clear();
	std::swap(twoViewMatches.matches, filtered_indices);
	log_message(conf, "View [" + util::string::get(viewId1) + ", " + util::string::get(viewId2) + "], raw matches count, filtered matches count: " + util::string::get(unfiltered_matches.size()) + ", " + util::string::get(twoViewMatches.matches.size()));
	pairwise_matching->push_back(twoViewMatches);
	return true;
}

bool two_view_matching(AppSettings const& conf, sfm::bundler::ViewportList* viewports, int viewId1, int viewId2, Mat& descImg1, Mat& descImg2,
	sfm::bundler::Matching &matcher, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> akaze_matches;

	// 	descriptorMatcher->match(descImg1, descImg2, akaze_matches, Mat());


	vector<vector<DMatch> > knn_matches;

	descriptorMatcher->knnMatch(descImg1, descImg2, knn_matches, 2);

	for (int i = 0; i < knn_matches.size(); i++) {
		DMatch _m;
		if (knn_matches[i].size() == 1) {
			_m = knn_matches[i][0];
		}
		else if (knn_matches[i].size()>1) {
			if (knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
				_m = knn_matches[i][0];
			}
			else {
				continue; // did not pass ratio test
			}
		}
		else {
			continue; // no match
		}

		akaze_matches.push_back(_m);
	}


	sfm::bundler::TwoViewMatching twoViewMatch;

	return filteringMatches(conf, viewports, viewId1, viewId2, akaze_matches, matcher, pairwise_matching);
}

void computingMatches(AppSettings const& conf, sfm::bundler::ViewportList* viewports, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors,
	sfm::bundler::Matching &matcher, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	size_t num_viewports = viewports->size();
	size_t num_pairs = num_viewports * (num_viewports - 1) / 2;
	size_t num_done = 0;
	for (size_t i = 0; i < num_pairs; ++i)
	{
		num_done += 1;
		int const view_1_id = (int)(0.5 + std::sqrt(0.25 + 2.0 * i));
		int const view_2_id = (int)i - view_1_id * (view_1_id - 1) / 2;
		vector<KeyPoint>& keypoints1 = keyPoints[view_1_id];
		vector<KeyPoint>& keypoints2 = keyPoints[view_2_id];
		if (keypoints1.size() == 0 || keypoints2.size() == 0)
			continue;

		two_view_matching(conf, viewports, view_1_id, view_2_id, descriptors[view_1_id], descriptors[view_2_id], matcher, pairwise_matching);
	}
}
/*2022.07.22*/
bool integratingMatches(const AppSettings& conf, sfm::bundler::ViewportList& new_viewports, sfm::bundler::PairwiseMatching& new_pairwise_matching,
	sfm::bundler::ViewportList& dst_viewports, sfm::bundler::PairwiseMatching& dst_pairwise_matching)
{
	if (new_viewports.size() != dst_viewports.size())
	{
		log_message(conf, "cv_matcher error:new view ports count is not equal to dst view ports count.");
		return false;
	}
	for (int i = 0; i < new_pairwise_matching.size(); ++i)
	{
		int new_view_1_id = new_pairwise_matching.at(i).view_1_id;
		int new_view_2_id = new_pairwise_matching.at(i).view_2_id;
		int offset1 = dst_viewports.at(new_view_1_id).features.positions.size();
		int offset2 = dst_viewports.at(new_view_2_id).features.positions.size();
		sfm::CorrespondenceIndices& new_matches = new_pairwise_matching.at(i).matches;
		for (int j = 0; j < new_matches.size(); ++j)
		{
			new_matches.at(j).first += offset1;
			new_matches.at(j).second += offset2;
		}
		dst_pairwise_matching.push_back(new_pairwise_matching.at(i));
	}
	for (int i = 0; i < new_viewports.size(); ++i)
	{
		const mvs::PixelCoords& new_feature_positions = new_viewports[i].features.positions;
		mvs::PixelCoords& dst_feature_positions = dst_viewports[i].features.positions;
		for (int j = 0; j < new_feature_positions.size(); ++j)
		{
			dst_feature_positions.push_back(new_feature_positions.at(j));
		}
	}
	return true;
}
/**/
void computingPairwiseMatching(const AppSettings& conf, sfm::bundler::ViewportList* viewports, vector<Mat> &imgs,
	Ptr<Feature2D> b, sfm::bundler::Matching &matcher, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	vector<vector<KeyPoint>> keyPointLists;
	vector<Mat> descriptors;
	keyPointLists.resize(imgs.size());
	descriptors.resize(imgs.size());
	viewports->resize(imgs.size());
	for (int i = 0; i < imgs.size(); ++i)
	{
		keyPointLists[i].clear();
		b->detectAndCompute(imgs[i], Mat(), keyPointLists[i], descriptors[i], false);
		viewports->at(i).features.width = imgs[i].cols;
		viewports->at(i).features.height = imgs[i].rows;
		float const fwidth = static_cast<float>(viewports->at(i).features.width);
		float const fheight = static_cast<float>(viewports->at(i).features.height);
		float const fnorm = std::max(fwidth, fheight);
		viewports->at(i).features.positions.resize(keyPointLists[i].size());
		mvs::PixelCoords& positions = viewports->at(i).features.positions;
		for (int j = 0; j < keyPointLists[i].size(); ++j)
		{
			math::Vec2f pos;
			pos[0] = keyPointLists[i].at(j).pt.x;
			pos[1] = keyPointLists[i].at(j).pt.y;
			positions[j][0] = (pos[0] + 0.5f - fwidth / 2.0f) / fnorm;
			positions[j][1] = (pos[1] + 0.5f - fheight / 2.0f) / fnorm;
		}
	}
	computingMatches(conf, viewports, keyPointLists, descriptors, matcher, pairwise_matching);
}

void computingPairwiseMatching(const AppSettings& conf, sfm::bundler::ViewportList* viewports, vector<string> &img_paths,
	Ptr<Feature2D> b, sfm::bundler::Matching &matcher, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	int img_count = img_paths.size();
	vector<vector<KeyPoint>> keyPointLists;
	vector<Mat> descriptors;
	keyPointLists.resize(img_count);
	descriptors.resize(img_count);
	viewports->resize(img_count);
	for (int i = 0; i < img_count; ++i)
	{
		keyPointLists[i].clear();
		auto img = imread(img_paths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		b->detectAndCompute(img, Mat(), keyPointLists[i], descriptors[i], false);
		viewports->at(i).features.width = img.cols;
		viewports->at(i).features.height = img.rows;
		float const fwidth = static_cast<float>(viewports->at(i).features.width);
		float const fheight = static_cast<float>(viewports->at(i).features.height);
		float const fnorm = std::max(fwidth, fheight);
		viewports->at(i).features.positions.resize(keyPointLists[i].size());
		mvs::PixelCoords& positions = viewports->at(i).features.positions;
		for (int j = 0; j < keyPointLists[i].size(); ++j)
		{
			math::Vec2f pos;
			pos[0] = keyPointLists[i].at(j).pt.x;
			pos[1] = keyPointLists[i].at(j).pt.y;
			positions[j][0] = (pos[0] + 0.5f - fwidth / 2.0f) / fnorm;
			positions[j][1] = (pos[1] + 0.5f - fheight / 2.0f) / fnorm;
		}
	}
	computingMatches(conf, viewports, keyPointLists, descriptors, matcher, pairwise_matching);
}

void features_and_matching(mve::Scene::Ptr scene, AppSettings const& conf, sfm::bundler::ViewportList* viewports, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	/* Feature computation for the scene. */
	sfm::bundler::Features::Options feature_opts;
	feature_opts.image_embedding = conf.sfmSettings.original_name;
	feature_opts.max_image_size = conf.sfmSettings.max_image_size;
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
	matching_opts.use_lowres_matching = conf.sfmSettings.lowres_matching;
	matching_opts.match_num_previous_frames = conf.sfmSettings.video_matching;
	matching_opts.matcher_type = conf.sfmSettings.cascade_hashing
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

/*2022.07.22*/
void cv_multi_feature_matching(mve::Scene::Ptr scene, AppSettings const& conf, sfm::bundler::Matching &matcher,
	sfm::bundler::ViewportList* final_viewports,
	sfm::bundler::PairwiseMatching* final_pairwise_matching)
{
	vector<Mat> imgs;
	mve::Scene::ViewList const& views = scene->get_views();
	int view_count = views.size();
	imgs.clear();
	std::string img_ext = conf.sceneSettings.img_extension;
	std::vector<string> img_paths;
	img_paths.resize(0);
	for (int i = 0; i < view_count; ++i)
	{
		string path = views.at(i)->get_directory() + "/original." + img_ext;
		auto img = imread(path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
		imgs.push_back(img);
		img_paths.push_back(path);
	}
	int nFeatureTypes = 3;
	Ptr<Feature2D> b[3];
	b[0] = SIFT::create(0, 3, 0.02f);
	b[1] = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 0, 3, 0.00035f);
	b[2] = ORB::create();
	final_viewports->resize(view_count);
	final_pairwise_matching->clear();
	if (conf.sfmSettings.match_mode == 0)
	{
		for (int i = 0; i < 2; ++i)
		{
			sfm::bundler::ViewportList viewports;
			sfm::bundler::PairwiseMatching pairwise_matching;
			computingPairwiseMatching(conf, &viewports, img_paths, b[i], matcher, &pairwise_matching);
			//computingPairwiseMatching(conf, &viewports, imgs, b[i], matcher, &pairwise_matching);
			integratingMatches(conf, viewports, pairwise_matching, *final_viewports, *final_pairwise_matching);
		}
	}
	if (conf.sfmSettings.match_mode == 1)
	{
		sfm::bundler::ViewportList viewports;
		sfm::bundler::PairwiseMatching pairwise_matching;
		computingPairwiseMatching(conf, &viewports, imgs, b[1], matcher, &pairwise_matching);
		integratingMatches(conf, viewports, pairwise_matching, *final_viewports, *final_pairwise_matching);
	}
	if (conf.sfmSettings.match_mode == 2)
	{
		sfm::bundler::ViewportList viewports;
		sfm::bundler::PairwiseMatching pairwise_matching;
		computingPairwiseMatching(conf, &viewports, imgs, b[0], matcher, &pairwise_matching);
		integratingMatches(conf, viewports, pairwise_matching, *final_viewports, *final_pairwise_matching);
	}
}

void cv_features_matching(mve::Scene::Ptr scene, AppSettings const& conf, sfm::bundler::ViewportList* viewports,
	sfm::bundler::PairwiseMatching* pairwise_matching)
{
	sfm::bundler::Matching::Options matching_opts;
	//matching_opts.ransac_opts.max_iterations = 1000;
	//matching_opts.ransac_opts.threshold = 0.0015;
	matching_opts.ransac_opts.verbose_output = false;
	matching_opts.use_lowres_matching = conf.sfmSettings.lowres_matching;
	matching_opts.match_num_previous_frames = conf.sfmSettings.video_matching;
	matching_opts.matcher_type = conf.sfmSettings.cascade_hashing
		? sfm::bundler::Matching::MATCHER_CASCADE_HASHING
		: sfm::bundler::Matching::MATCHER_EXHAUSTIVE;
	sfm::bundler::Matching bundler_matching(matching_opts);
	cv_multi_feature_matching(scene, conf, bundler_matching, viewports, pairwise_matching);
}

void check_prebundle(AppSettings const& conf)
{
	std::string const prebundle_path = util::fs::join_path(conf.sceneSettings.path_scene, conf.sfmSettings.prebundle_file);

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
		scene = mve::Scene::create(conf.sceneSettings.path_scene);
	}
	catch (std::exception& e) {
		std::cerr << "Error loading scene: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	sfm::bundler::SurveyPointList survey;
	if (!conf.sfmSettings.survey_file.empty())
		sfm::bundler::load_survey_from_file(conf.sfmSettings.survey_file, &survey);

	/* Try to load the pairwise matching from the prebundle. */
	std::string const prebundle_path = util::fs::join_path(scene->get_path(), conf.sfmSettings.prebundle_file);
	sfm::bundler::ViewportList viewports;
	sfm::bundler::PairwiseMatching pairwise_matching;
	g_p3DProgressCallback(18, 10, "features and matching in sfm recon starts.");
	if (!util::fs::file_exists(prebundle_path.c_str())) {
		log_message(conf, "Starting feature matching.");
		util::system::rand_seed(RAND_SEED_MATCHING);
		if (conf.sfmSettings.match_mode == 3)
		{
			log_message(conf, "Starting mve feature matching.");
			features_and_matching(scene, conf, &viewports, &pairwise_matching);
		}
		else
		{
			log_message(conf, "Starting cv feature matching.");
			cv_features_matching(scene, conf, &viewports, &pairwise_matching);
		}

		// 		std::cout << "Saving pre-bundle to file..." << std::endl;
		log_message(conf, "Saving pre-bundle to file...");

		sfm::bundler::save_prebundle_to_file(viewports, pairwise_matching, prebundle_path);
	}
	else if (!conf.sfmSettings.skip_sfm) {
		// 		std::cout << "Loading pairwise matching from file..." << std::endl;
		log_message(conf, "Loading pairwise matching from file...");
		sfm::bundler::load_prebundle_from_file(prebundle_path, &viewports, &pairwise_matching);
	}
	g_p3DProgressCallback(25, 20, "features and matching in sfm recon done.");
	if (conf.sfmSettings.skip_sfm) {
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
		if (conf.sfmSettings.intrinsics_from_views)
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
	g_p3DProgressCallback(28, 50, "incremental SfM starts.");
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
	g_p3DProgressCallback(30, 60, "tracking features done.");
	/* Remove color data and pairwise matching to save memory. */
	for (std::size_t i = 0; i < viewports.size(); ++i)
		viewports[i].features.colors.clear();
	pairwise_matching.clear();

	/* Search for a good initial pair, or use the user-specified one. */
	sfm::bundler::InitialPair::Result init_pair_result;
	sfm::bundler::InitialPair::Options init_pair_opts;
	if (conf.sfmSettings.initial_pair_1 < 0 || conf.sfmSettings.initial_pair_2 < 0)
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
		init_pair.compute_pair(conf.sfmSettings.initial_pair_1, conf.sfmSettings.initial_pair_2,
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
	g_p3DProgressCallback(32, 70, "computing init pair done.");
	std::cout << "Using views " << init_pair_result.view_1_id
		<< " and " << init_pair_result.view_2_id
		<< " as initial pair." << std::endl;

	/* Incrementally compute full bundle. */
	sfm::bundler::Incremental::Options incremental_opts;
	//incremental_opts.pose_p3p_opts.max_iterations = 1000;
	//incremental_opts.pose_p3p_opts.threshold = 0.005f;
	incremental_opts.pose_p3p_opts.verbose_output = false;
	incremental_opts.track_error_threshold_factor = conf.sfmSettings.track_error_thres_factor;
	incremental_opts.new_track_error_threshold = conf.sfmSettings.new_track_error_thres;
	incremental_opts.min_triangulation_angle = MATH_DEG2RAD(1.0);
	incremental_opts.ba_fixed_intrinsics = conf.sfmSettings.fixed_intrinsics;
	//incremental_opts.ba_shared_intrinsics = conf.shared_intrinsics;
	incremental_opts.verbose_output = true;
	incremental_opts.verbose_ba = conf.sfmSettings.verbose_ba;

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
	g_p3DProgressCallback(33, 80, "bundle adjustment done.");
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
				incremental.triangulate_new_tracks(conf.sfmSettings.min_views_per_track);
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
		int const full_ba_skip_views = conf.sfmSettings.always_full_ba ? 0
			: std::min(100, num_cameras_reconstructed / 10);
		if (full_ba_num_skipped < full_ba_skip_views)
		{
			std::cout << "Skipping full bundle adjustment (skipping "
				<< full_ba_skip_views << " views)." << std::endl;
			full_ba_num_skipped += 1;
		}
		else
		{
			incremental.triangulate_new_tracks(conf.sfmSettings.min_views_per_track);
			std::cout << "Running full bundle adjustment..." << std::endl;
			incremental.bundle_adjustment_full();
			incremental.invalidate_large_error_tracks();
			full_ba_num_skipped = 0;
		}
	}

	std::cout << "SfM reconstruction took " << timer.get_elapsed() << " ms." << std::endl;
	log_message(conf, "SfM reconstruction took " + util::string::get(timer.get_elapsed()) + "ms.");

	/* Normalize scene if requested. */
	if (conf.sfmSettings.normalize_scene)
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
		if (!conf.sfmSettings.undistorted_name.empty())
		{
			mve::ByteImage::Ptr original
				= view->get_byte_image(conf.sfmSettings.original_name);
			if (original == nullptr)
				continue;
			mve::ByteImage::Ptr undist
				= mve::image::image_undistort_k2k4<uint8_t>
				(original, cam.flen, cam.dist[0], cam.dist[1]);
			view->set_image(undist, conf.sfmSettings.undistorted_name);
		}

#pragma omp critical
		std::cout << "Saving view " << view->get_directory() << std::endl;
		view->save_view();
		view->cache_cleanup();
	}
	log_message(conf, "SfM reconstruction done.\n");
}

