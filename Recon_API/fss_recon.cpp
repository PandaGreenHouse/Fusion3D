#include "common.h"

int fssrecon(AppSettings& conf/*, fssr::SampleIO::Options const& pset_opts*/)
{
	log_message(conf, "Floating Scale Surface Reconstruction starts.");

	// 	util::system::register_segfault_handler();
	// 	util::system::print_build_timestamp("Floating Scale Surface Reconstruction");

	/* Init default settings. */
	fssr::SampleIO::Options pset_opts;
	std::string in_mesh = conf.psetSettings.pset_name1;//util::fs::join_path(app_opts.path_scene, "pset-L2.ply");////
	std::string out_mesh = util::fs::join_path(conf.sceneSettings.path_scene, "surface-L2.ply");
	conf.FssreconSettings.in_files.push_back(in_mesh);
	conf.FssreconSettings.in_files.push_back(out_mesh);
	if (conf.FssreconSettings.in_files.size() < 2)
	{
		return EXIT_FAILURE;
	}
	conf.FssreconSettings.out_mesh = conf.FssreconSettings.in_files.back();
	conf.FssreconSettings.in_files.pop_back();

	if (conf.FssreconSettings.refine_octree < 0 || conf.FssreconSettings.refine_octree > 3)
	{
		// 		std::cerr << "Unreasonable refine level of " << conf.refine_octree << ", exiting." << std::endl;
		log_message(conf, "Unreasonable refine level of " + util::string::get(conf.FssreconSettings.refine_octree) + ", exiting.");
		return EXIT_FAILURE;
	}

	/* Load input point set and insert samples in the octree. */
	fssr::IsoOctree octree;
	for (std::size_t i = 0; i < conf.FssreconSettings.in_files.size(); ++i)
	{
		g_p3DProgressCallback(70, 80*(float)i/ conf.FssreconSettings.in_files.size(), "meshing starts.");
		// 		std::cout << "Loading: " << conf.in_files[i] << "..." << std::endl;
		log_message(conf, "Loading: " + util::string::get(conf.FssreconSettings.in_files[i]) + "...");
		util::WallTimer timer;

		fssr::SampleIO loader(pset_opts);
		loader.open_file(conf.FssreconSettings.in_files[i]);
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
	if (conf.FssreconSettings.refine_octree > 0)
	{
		std::cout << "Refining octree..." << std::flush;
		util::WallTimer timer;
		for (int i = 0; i < conf.FssreconSettings.refine_octree; ++i) {
			octree.refine_octree();
		}
		std::cout << " took " << timer.get_elapsed() << "ms" << std::endl;
	}

	/* Compute voxels. */
	octree.limit_octree_level();
	octree.print_stats(std::cout);
	octree.compute_voxels();
	octree.clear_samples();
	g_p3DProgressCallback(75, 85, "extracting iso surface starts.");
	/* Extract isosurface. */
	mve::TriangleMesh::Ptr mesh;
	{
		// 		std::cout << "Extracting isosurface..." << std::endl;
		log_message(conf, "Extracting isosurface...");

		util::WallTimer timer;
		fssr::IsoSurface iso_surface(&octree, conf.FssreconSettings.interp_type);
		mesh = iso_surface.extract_mesh();

		// 		std::cout << "  Done. Surface extraction took " << timer.get_elapsed() << "ms." << std::endl;
		log_message(conf, "Done. Surface extraction took " + util::string::get(timer.get_elapsed()) + "ms.");
	}
	octree.clear();
	g_p3DProgressCallback(80, 90, "extracting iso surface done.");
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
	mve::geom::save_ply_mesh(mesh, conf.FssreconSettings.out_mesh, ply_opts);

	// 	std::cout << "All done. Remember to clean the output mesh." << std::endl;
	log_message(conf, "All done. Remember to clean the output mesh.");
	log_message(conf, "Floating Scale Surface Reconstruction ends.");

	return EXIT_SUCCESS;
}