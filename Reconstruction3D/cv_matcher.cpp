#include "cv_matcher.h"

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

void cv_features_and_matching(mve::Scene::Ptr scene, std::string &img_extension, sfm::bundler::Matching &matcher,
							  sfm::bundler::ViewportList* viewports, sfm::bundler::PairwiseMatching* pairwise_matching)
{
	mve::Scene::ViewList const& views = scene->get_views();
	int view_count = views.size();
	viewports->clear();
	viewports->resize(view_count);
	Ptr<Feature2D> b[3];
	b[0] = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 0, 3, 0.00035f);
	b[1] = SIFT::create(0, 3, 0.02f);
	b[2] = ORB::create();
	vector<vector<vector<KeyPoint>>> keyPointLists;
	vector<vector<Mat>> descriptors;
	keyPointLists.resize(view_count);
	descriptors.resize(view_count);
	for (int i = 0; i < view_count; ++i)
	{
		keyPointLists[i].resize(3);
		descriptors[i].resize(3);
		std::string path = views.at(i)->get_directory() + "/original." + img_extension;
		auto img = imread(path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
		for (int j = 0; j < 3; ++j)
		{
			keyPointLists[i][j].clear();
			b[j]->detectAndCompute(img, Mat(), keyPointLists[i][j], descriptors[i][j], false);
		}
	}
	/*mve::Scene::ViewList const& views = scene->get_views();
	int view_count = views.size();
	/*Initialize viewports. */
	/*viewports->clear();
	viewports->resize(view_count);
	//Ptr<Feature2D> b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 0, 3, 0.00035f);//(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.00025f, 5, 5);AKAZE::DESCRIPTOR_KAZE
// 	Ptr<Feature2D> b = SIFT::create(0, 3, 0.02);
	Ptr<Feature2D> b[3];
	b[0] = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 0, 3, 0.00035f);
	b[1] = SIFT::create(0, 3, 0.02f);
	b[2] = ORB::create();
	vector<vector<KeyPoint>> keyPointLists;
	vector<Mat> descriptors;
	keyPointLists.resize(view_count);
	descriptors.resize(view_count);
	//Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	for (int i = 0; i < view_count; ++i)
	{
		std::string path = views[i]->get_directory() + "/" + "original.jpg";
		keyPointLists[i].clear();
		auto img = imread(path, IMREAD_ANYCOLOR|IMREAD_ANYDEPTH);
		for (int j = 0; j < 3; ++j)
		{
			b[j]->detectAndCompute(img, Mat(), keyPointLists[i], descriptors[i], false);
		}
		
// 		std::cout << "key point count:" << keyPointLists[i].size() << std::endl;
		log_message(conf, "View " + util::string::get(i) + ", key point count : " + util::string::get(keyPointLists[i].size()));

		viewports->at(i).features.positions.resize(keyPointLists[i].size());
		viewports->at(i).features.width = img.cols;
		viewports->at(i).features.height = img.rows;
		/* Normalize image coordinates. */
		/*float const fwidth = static_cast<float>(viewports->at(i).features.width);
		float const fheight = static_cast<float>(viewports->at(i).features.height);
		float const fnorm = std::max(fwidth, fheight);
		for (int j = 0; j < keyPointLists[i].size(); ++j)
		{
			math::Vec2f pos;
			pos[0] = keyPointLists[i][j].pt.x;
			pos[1] = keyPointLists[i][j].pt.y;
			viewports->at(i).features.positions[j][0] = (pos[0] + 0.5f - fwidth / 2.0f) / fnorm;
			viewports->at(i).features.positions[j][1] = (pos[1] + 0.5f - fheight / 2.0f) / fnorm;
		}
	}
	pairwise_matching->clear();

	computingMatches(conf, viewports, keyPointLists, descriptors, matcher, pairwise_matching);*/
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
void computingPairwiseMatching(const AppSettings& conf, sfm::bundler::ViewportList* viewports,	vector<Mat> &imgs, 
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
/*2022.07.22*/
void cv_multi_feature_matching(mve::Scene::Ptr scene, AppSettings const& conf, std::string &extension, sfm::bundler::Matching &matcher,
							   sfm::bundler::ViewportList* final_viewports,
							   sfm::bundler::PairwiseMatching* final_pairwise_matching)
{
	vector<Mat> imgs;
	mve::Scene::ViewList const& views = scene->get_views();
	int view_count = views.size();
	imgs.clear();
	for (int i = 0; i < view_count; ++i)
	{
		string path = views.at(i)->get_directory() + "/original." + extension;
		auto img = imread(path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
		imgs.push_back(img);
	}
	int nFeatureTypes = 3;
	Ptr<Feature2D> b[3];
	b[0] = SIFT::create(0, 3, 0.02f);
	b[1] = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 0, 3, 0.00035f);
	b[2] = ORB::create();
	final_viewports->resize(view_count);
	final_pairwise_matching->clear();
	//for (int i = 0; i < 2; ++i)
	{
		sfm::bundler::ViewportList viewports;
		sfm::bundler::PairwiseMatching pairwise_matching;
		computingPairwiseMatching(conf, &viewports, imgs, b[1], matcher, &pairwise_matching);
		integratingMatches(conf, viewports, pairwise_matching, *final_viewports, *final_pairwise_matching);
	}
}


