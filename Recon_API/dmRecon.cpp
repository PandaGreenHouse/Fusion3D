#include "common.h"
FancyProgressPrinter fancyProgressPrinter;
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
	if (width * height <= app_settings.sceneSettings.max_pixels)
		return 0;

	float const ratio = width * height / static_cast<float>(app_settings.sceneSettings.max_pixels);
	float const scale = std::ceil(std::log(ratio) / std::log(4.0f));

	std::cout << "Setting scale " << scale << " for " << width << "x" << height << " image." << std::endl;

	return std::max(0, static_cast<int>(scale));
}

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

int dmrecon(AppSettings& conf)
{
	log_message(conf, "Depthmap reconstruction starts.");

	// 	util::system::register_segfault_handler();
	// 	util::system::print_build_timestamp("MVE Depth Map Reconstruction");


	conf.dmSettings.mvs.useColorScale = true;// false;
	conf.sceneSettings.max_pixels = 0;
	conf.dmSettings.mvs.scale = 2;//3
	conf.dmSettings.mvs.keepDzMap = true;
	conf.dmSettings.mvs.keepConfidenceMap = true;
	conf.dmSettings.write_ply = false;// true;
	conf.dmSettings.progress_style = PROGRESS_SIMPLE; //PROGRESS_SILENT;
	conf.dmSettings.force_recon = false;// true;

							 /* don't show progress twice */
	if (conf.dmSettings.progress_style != PROGRESS_SIMPLE) {
		conf.dmSettings.mvs.quiet = true;
	}


	/* Load MVE scene. */
	mve::Scene::Ptr scene;
	try {
		scene = mve::Scene::create(conf.sceneSettings.path_scene);
		scene->get_bundle();
	}
	catch (std::exception& e) {
		// 		std::cerr << "Error loading scene: " << e.what() << std::endl;
		log_message(conf, "Error loading scene: " + util::string::get(e.what()));

		return EXIT_FAILURE;
	}

	/* Settings for Multi-view stereo */
	conf.dmSettings.mvs.writePlyFile = conf.dmSettings.write_ply;
	conf.dmSettings.mvs.plyPath = util::fs::join_path(conf.sceneSettings.path_scene, conf.dmSettings.ply_dest);

	fancyProgressPrinter.setBasePath(conf.sceneSettings.path_scene);
	fancyProgressPrinter.setNumViews(scene->get_views().size());
	if (conf.dmSettings.progress_style == PROGRESS_FANCY) {
		fancyProgressPrinter.start();
	}

	util::WallTimer timer;
	if (conf.dmSettings.master_id >= 0) {
		/* Calculate scale from max pixels. */
		if (conf.sceneSettings.max_pixels > 0) {
			conf.dmSettings.mvs.scale = get_scale_from_max_pixels(scene, conf, conf.dmSettings.mvs);
		}

		// 		std::cout << "Reconstructing view ID " << conf.master_id << std::endl;
		log_message(conf, "Reconstructing view ID " + util::string::get(conf.dmSettings.master_id));

		conf.dmSettings.mvs.refViewNr = (std::size_t)conf.dmSettings.master_id;
		fancyProgressPrinter.addRefView(conf.dmSettings.master_id);
		try {
			reconstruct(scene, conf.dmSettings.mvs);
		}
		catch (std::exception &err) {
			// 			std::cerr << err.what() << std::endl;
			log_message(conf, "Error reconstruct: " + util::string::get(err.what()));
			return EXIT_FAILURE;
		}
	}
	else {
		mve::Scene::ViewList& views(scene->get_views());
		if (conf.dmSettings.view_ids.empty()) {
			// 			std::cout << "Reconstructing all views..." << std::endl;
			log_message(conf, "Reconstructing all views...");
			for (std::size_t i = 0; i < views.size(); ++i) {
				conf.dmSettings.view_ids.push_back(i);
			}
		}
		else {
			// 			std::cout << "Reconstructing views from list..." << std::endl;
			log_message(conf, "Reconstructing views from list...");
		}
		fancyProgressPrinter.addRefViews(conf.dmSettings.view_ids);

#pragma omp parallel for schedule(dynamic, 1)
#if !defined(_MSC_VER)
		for (std::size_t i = 0; i < conf.view_ids.size(); ++i)
#else
		for (int64_t i = 0; i < conf.dmSettings.view_ids.size(); ++i)
#endif
		{
			std::size_t id = conf.dmSettings.view_ids[i];
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
			mvs::Settings settings(conf.dmSettings.mvs);
			settings.refViewNr = id;
			if (conf.sceneSettings.max_pixels > 0) {
				settings.scale = get_scale_from_max_pixels(scene, conf, settings);
			}

			std::string embedding_name = "depth-L" + util::string::get(settings.scale);
			if (!conf.dmSettings.force_recon && views[id]->has_image(embedding_name)) {
				continue;
			}
			g_p3DProgressCallback(33, 100*(float)i/ conf.dmSettings.view_ids.size(), "dm recon is in progress.");
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

	if (conf.dmSettings.progress_style == PROGRESS_FANCY) {
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
