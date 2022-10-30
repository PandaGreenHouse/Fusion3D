#include "common.h"

void reconstruct_sgm_depth_for_view(AppSettings const& conf,
	smvs::StereoView::Ptr main_view,
	std::vector<smvs::StereoView::Ptr> neighbors,
	mve::Bundle::ConstPtr bundle = nullptr)
{
	smvs::SGMStereo::Options sgm_opts;
	sgm_opts.scale = conf.smvsSettings.sgm_scale;
	sgm_opts.num_steps = 128;
	sgm_opts.debug_lvl = conf.smvsSettings.debug_lvl;
	sgm_opts.min_depth = conf.smvsSettings.sgm_min;
	sgm_opts.max_depth = conf.smvsSettings.sgm_max;

	util::WallTimer sgm_timer;
	mve::FloatImage::Ptr d1 = smvs::SGMStereo::reconstruct(sgm_opts, main_view,	neighbors[0], bundle);
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

	if (conf.smvsSettings.debug_lvl > 0)
		std::cout << "SGM took: " << sgm_timer.get_elapsed_sec()
		<< "sec" << std::endl;

	main_view->write_depth_to_view(d1, "smvs-sgm");
}

void generate_mesh(AppSettings& conf, mve::Scene::Ptr scene,
	std::string const& input_name, std::string const& dm_name)
{
	std::cout << "Generating ";
	if (conf.smvsSettings.create_triangle_mesh)
		std::cout << "Mesh";
	else
		std::cout << "Pointcloud";
	if (conf.smvsSettings.cut_surface)
		std::cout << ", Cutting surfaces";

	util::WallTimer timer;
	mve::Scene::ViewList recon_views;
	for (int i : conf.dmSettings.view_ids)
		recon_views.push_back(scene->get_views()[i]);

	std::cout << " for " << recon_views.size() << " views ..." << std::endl;

	smvs::MeshGenerator::Options meshgen_opts;
	meshgen_opts.num_threads = conf.smvsSettings.num_threads;
	meshgen_opts.cut_surfaces = conf.smvsSettings.cut_surface;
	meshgen_opts.simplify = conf.smvsSettings.simplify;
	meshgen_opts.create_triangle_mesh = conf.smvsSettings.create_triangle_mesh;

	smvs::MeshGenerator meshgen(meshgen_opts);
	mve::TriangleMesh::Ptr mesh = meshgen.generate_mesh(recon_views,
		input_name, dm_name);

	if (conf.smvsSettings.aabb_string.size() > 0)
	{
		std::cout << "Clipping to AABB: (" << conf.smvsSettings.aabb_min << ") / ("
			<< conf.smvsSettings.aabb_max << ")" << std::endl;

		mve::TriangleMesh::VertexList const& verts = mesh->get_vertices();
		std::vector<bool> aabb_clip(verts.size(), false);
		for (std::size_t v = 0; v < verts.size(); ++v)
			for (int i = 0; i < 3; ++i)
				if (verts[v][i] < conf.smvsSettings.aabb_min[i]
					|| verts[v][i] > conf.smvsSettings.aabb_max[i])
					aabb_clip[v] = true;
		mesh->delete_vertices_fix_faces(aabb_clip);
	}

	std::cout << "Done. Took: " << timer.get_elapsed_sec() << "s" << std::endl;

	if (conf.smvsSettings.create_triangle_mesh)
		mesh->recalc_normals();

	/* Build mesh name */
	std::string meshname = "smvs-";
	if (conf.smvsSettings.create_triangle_mesh)
		meshname += "m-";
	if (conf.smvsSettings.use_shading)
		meshname += "S";
	else
		meshname += "B";
	meshname += util::string::get(conf.smvsSettings.input_scale) + ".ply";
	conf.psetSettings.dmname = meshname;
	conf.psetSettings.pset_name = util::fs::join_path(conf.sceneSettings.path_scene, conf.psetSettings.dmname);

	meshname = util::fs::join_path(scene->get_path(), meshname);
	mve::Scene::ViewList& views = scene->get_views();
	/* Save mesh */
	mve::geom::SavePLYOptions opts;
	opts.write_vertex_normals = true;
	opts.write_vertex_values = true;
	opts.write_vertex_confidences = true;
	mve::geom::save_ply_mesh(mesh, meshname, opts);
	log_message(conf, "smvs's pset is saved.");
	
	/*added for extracting object 2022/08/03*/
	log_message(conf, "extracting object...");
	log_message(conf, "obj path: " + conf.psetSettings.pset_name1);
	mve::TriangleMesh::Ptr pset_obj(mve::TriangleMesh::create());
	extractObj(scene, conf.plane_tolerance, mesh, pset_obj);
	log_message(conf, "count of vertices extracted: "+util::string::get(pset_obj->get_vertices().size()));
	mve::geom::save_ply_mesh(pset_obj, conf.psetSettings.pset_name1, opts);
	log_message(conf, "object is saved.");
}

int smvsrecon(AppSettings& conf)
{
	log_message(conf, "Shading-aware Multi-view Stereo starts.");

	// 	util::system::register_segfault_handler();
	// 	util::system::print_build_timestamp("Shading-aware Multi-view Stereo");

	/* Start processing */

	/* Load scene */
	mve::Scene::Ptr scene = mve::Scene::create(conf.sceneSettings.path_scene);
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

		conf.smvsSettings.use_sgm = true;
		if (conf.smvsSettings.sgm_max == 0.0)
		{
			std::cout << "Error: No bundle file and SGM depth given, "
				"please use the --sgm-range option to specify the "
				"depth sweep range for SGM." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	log_message(conf, "smvs-input scale:" + util::string::get(conf.smvsSettings.input_scale) + "smvs-output scale:" + util::string::get(conf.smvsSettings.output_scale));
	/* Reconstruct all views if no specific list is given */
	if (conf.dmSettings.view_ids.empty()) {
		for (auto view : views) {
			if (view != nullptr && view->is_camera_valid()) {
				conf.dmSettings.view_ids.push_back(view->get_id());
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
	g_p3DProgressCallback(40, 20, "smvs recon is in progress1.");
	if (conf.smvsSettings.clean_scene)
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
	g_p3DProgressCallback(42, 25, "smvs recon is in progress2.");
	/* Scale input images */
	if (conf.smvsSettings.input_scale < 0)
	{
		double avg_image_size = 0;
		int view_counter = 0;
		for (std::size_t i = 0; i < views.size(); ++i)
		{
			mve::View::Ptr view = views[i];
			if (view == nullptr || !view->has_image(conf.smvsSettings.image_embedding)) {
				continue;
			}

			mve::View::ImageProxy const* proxy = view->get_image_proxy(conf.smvsSettings.image_embedding);
			uint32_t size = proxy->width * proxy->height;
			avg_image_size += static_cast<double>(size);
			view_counter += 1;
		}
		avg_image_size /= static_cast<double>(view_counter);

		if (avg_image_size > conf.sceneSettings.max_pixels) {
			conf.smvsSettings.input_scale = std::ceil(std::log2(avg_image_size / conf.sceneSettings.max_pixels) / 2);
		}
		else {
			conf.smvsSettings.input_scale = 0;
		}
		// 		std::cout << "Automatic input scale: " << conf.input_scale << std::endl;
		log_message(conf, "Automatic input scale: " + util::string::get(conf.smvsSettings.input_scale));
	}
	g_p3DProgressCallback(44, 28, "smvs recon is in progress3.");
	std::string input_name;
	if (conf.smvsSettings.input_scale > 0)
		input_name = "undist-L" + util::string::get(conf.smvsSettings.input_scale);
	else
		input_name = conf.smvsSettings.image_embedding;
	std::cout << "Input embedding: " << input_name << std::endl;

	std::string output_name;
	if (conf.smvsSettings.use_shading)
		output_name = "smvs-S" + util::string::get(conf.smvsSettings.input_scale);
	else
		output_name = "smvs-B" + util::string::get(conf.smvsSettings.input_scale);
	std::cout << "Output embedding: " << output_name << std::endl;
	conf.smvsSettings.out_pset_name = output_name;
	/* Clean view id list */
	std::vector<int> ignore_list;
	for (std::size_t v = 0; v < conf.dmSettings.view_ids.size(); ++v)
	{
		int const i = conf.dmSettings.view_ids[v];
		if (i > static_cast<int>(views.size() - 1) || views[i] == nullptr)
		{
			std::cout << "View ID " << i << " invalid, skipping view." << std::endl;
			ignore_list.push_back(i);
			continue;
		}
		if (!views[i]->has_image(conf.smvsSettings.image_embedding))
		{
			std::cout << "View ID " << i << " missing image embedding, " << "skipping view." << std::endl;
			ignore_list.push_back(i);
			continue;
		}
	}
	g_p3DProgressCallback(44, 30, "smvs recon is in progress4.");
	for (auto const& id : ignore_list)
		conf.dmSettings.view_ids.erase(std::remove(conf.dmSettings.view_ids.begin(),
			conf.dmSettings.view_ids.end(), id));
	g_p3DProgressCallback(45, 40, "smvs recon is in progress5.");
	/* Add views to reconstruction list */
	std::vector<int> reconstruction_list;
	int already_reconstructed = 0;
	for (std::size_t v = 0; v < conf.dmSettings.view_ids.size(); ++v)
	{
		int const i = conf.dmSettings.view_ids[v];
		/* Only reconstruct missing views or if forced */
		if (conf.dmSettings.force_recon || !views[i]->has_image(output_name))
			reconstruction_list.push_back(i);
		else
			already_reconstructed += 1;
	}
	g_p3DProgressCallback(48, 60, "smvs recon is in progress6.");
	if (already_reconstructed > 0)
		std::cout << "Skipping " << already_reconstructed << " views that are already reconstructed." << std::endl;

	/* Create reconstruction threads */
	ThreadPool thread_pool(std::max<std::size_t>(conf.smvsSettings.num_threads, 1));

	/* View selection */
	smvs::ViewSelection::Options view_select_opts;
	view_select_opts.num_neighbors = conf.smvsSettings.num_neighbors;
	view_select_opts.embedding = conf.smvsSettings.image_embedding;
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
	g_p3DProgressCallback(50, 80, "smvs recon is in progress7.");
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
		if (view_neighbors[v].size() < conf.smvsSettings.min_neighbors)
			skipped.push_back(reconstruction_list[v]);
		else
		{
			final_reconstruction_list.push_back(reconstruction_list[v]);
			final_view_neighbors.push_back(view_neighbors[v]);
		}
	g_p3DProgressCallback(55, 85, "smvs recon is in progress8.");
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
			|| !view->has_image(conf.smvsSettings.image_embedding)
			|| view->has_image(input_name))
			continue;

		resize_tasks.emplace_back(thread_pool.add_task(
			[view, &input_name, &conf]
		{
			mve::ByteImage::ConstPtr input =
				view->get_byte_image(conf.smvsSettings.image_embedding);
			mve::ByteImage::Ptr scld = input->duplicate();
			for (int i = 0; i < conf.smvsSettings.input_scale; ++i)
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
	g_p3DProgressCallback(60, 90, "smvs recon is in progress9.");
	std::vector<std::future<void>> results;
	std::mutex counter_mutex;
	std::size_t started = 0;
	std::size_t finished = 0;
	util::WallTimer timer;

	for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
	{
		g_p3DProgressCallback(50, 20 + 80 * (float)v / reconstruction_list.size(), "smvs recon is in progress.");
		int const i = reconstruction_list[v];

		results.emplace_back(thread_pool.add_task(
			[v, i, &views, &conf, &counter_mutex, &input_name, &output_name,
			&started, &finished, &reconstruction_list, &view_neighbors,
			bundle, scene]
		{
			smvs::StereoView::Ptr main_view = smvs::StereoView::create(
				views[i], input_name, conf.smvsSettings.use_shading,
				conf.smvsSettings.gamma_correction);
			mve::Scene::ViewList neighbors = view_neighbors[v];

			std::vector<smvs::StereoView::Ptr> stereo_views;

			std::unique_lock<std::mutex> lock(counter_mutex);
			std::cout << "\rStarting "
				<< ++started << "/" << reconstruction_list.size()
				<< " ID: " << i
				<< " Neighbors: ";
			for (std::size_t n = 0; n < conf.smvsSettings.num_neighbors
				&& n < neighbors.size(); ++n)
				std::cout << neighbors[n]->get_id() << " ";
			std::cout << std::endl;
			lock.unlock();

			for (std::size_t n = 0; n < conf.smvsSettings.num_neighbors
				&& n < neighbors.size(); ++n)
			{
				smvs::StereoView::Ptr sv = smvs::StereoView::create(
					neighbors[n], input_name);
				stereo_views.push_back(sv);
			}

			if (conf.smvsSettings.use_sgm)
			{
				int sgm_width = views[i]->get_image_proxy(input_name)->width;
				int sgm_height = views[i]->get_image_proxy(input_name)->height;
				for (int scale = 0; scale < conf.smvsSettings.sgm_scale; ++scale)
				{
					sgm_width = (sgm_width + 1) / 2;
					sgm_height = (sgm_height + 1) / 2;
				}
				if (conf.smvsSettings.force_sgm || !views[i]->has_image("smvs-sgm")
					|| views[i]->get_image_proxy("smvs-sgm")->width !=
					sgm_width
					|| views[i]->get_image_proxy("smvs-sgm")->height !=
					sgm_height)
					reconstruct_sgm_depth_for_view(conf, main_view, stereo_views, bundle);
			}

			smvs::DepthOptimizer::Options do_opts;
			do_opts.regularization = 0.01 * conf.smvsSettings.regularization;
			do_opts.num_iterations = 5;
			do_opts.debug_lvl = conf.smvsSettings.debug_lvl;
			do_opts.min_scale = conf.smvsSettings.output_scale;
			do_opts.use_shading = conf.smvsSettings.use_shading;
			do_opts.output_name = output_name;
			do_opts.use_sgm = conf.smvsSettings.use_sgm;
			do_opts.full_optimization = conf.smvsSettings.full_optimization;
			do_opts.light_surf_regularization = conf.smvsSettings.light_surf_regularization;

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
	g_p3DProgressCallback(65, 95, "smvs recon is in progress10.");
	/* Generate mesh */
	if (!conf.smvsSettings.recon_only) {
		generate_mesh(conf, scene, input_name, output_name);
	}
	g_p3DProgressCallback(70, 100, "smvs recon is done.");
	log_message(conf, "Shading-aware Multi-view Stereo ends.");
}
