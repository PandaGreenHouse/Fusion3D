#include "common.h"

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

int mesh_clean(AppSettings& conf)
{
	log_message(conf, "FSSR Mesh Cleaning starts.");

	// 	util::system::register_segfault_handler();
	// 	util::system::print_build_timestamp("MVE FSSR Mesh Cleaning");

	// 	conf.out_mesh_clean = ;

	conf.meshCleanSettings.conf_threshold = 10;
	conf.meshCleanSettings.clean_degenerated = false;
	conf.meshCleanSettings.delete_scale = true;
	conf.meshCleanSettings.delete_conf = true;
	conf.meshCleanSettings.delete_colors = false;//true
	std::string cleaned_mesh_file = util::fs::join_path(conf.sceneSettings.path_scene, "clean.ply");
	/* Load input mesh. */
	mve::TriangleMesh::Ptr mesh;
	try
	{
		std::cout << "Loading mesh: " << conf.FssreconSettings.out_mesh << std::endl;
		mesh = mve::geom::load_mesh(conf.FssreconSettings.out_mesh);
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

	if (!mesh->has_vertex_confidences() && conf.meshCleanSettings.conf_threshold > 0.0f)
	{
		std::cerr << "Error: Confidence cleanup requested, but mesh "
			"has no confidence values." << std::endl;
		return EXIT_FAILURE;
	}

	if (mesh->get_faces().empty()
		&& (conf.meshCleanSettings.clean_degenerated || conf.meshCleanSettings.component_size > 0))
	{
		std::cerr << "Error: Components/faces cleanup "
			"requested, but mesh has no faces." << std::endl;
		return EXIT_FAILURE;
	}

	/* Remove low-confidence geometry. */
	if (conf.meshCleanSettings.conf_percentile > 0.0f)
		conf.meshCleanSettings.conf_threshold = percentile(mesh->get_vertex_confidences(),
			conf.meshCleanSettings.conf_percentile);
	if (conf.meshCleanSettings.conf_threshold > 0.0f)
	{
		std::cout << "Removing low-confidence geometry (threshold "
			<< conf.meshCleanSettings.conf_threshold << ")..." << std::endl;
		std::size_t num_verts = mesh->get_vertices().size();
		remove_low_conf_vertices(mesh, conf.meshCleanSettings.conf_threshold);
		std::size_t new_num_verts = mesh->get_vertices().size();
		std::cout << "  Deleted " << (num_verts - new_num_verts)
			<< " low-confidence vertices." << std::endl;
	}

	/* Remove isolated components if requested. */
	if (conf.meshCleanSettings.component_size > 0)
	{
		std::cout << "Removing isolated components below "
			<< conf.meshCleanSettings.component_size << " vertices..." << std::endl;
		std::size_t num_verts = mesh->get_vertices().size();
		mve::geom::mesh_components(mesh, conf.meshCleanSettings.component_size);
		std::size_t new_num_verts = mesh->get_vertices().size();
		std::cout << "  Deleted " << (num_verts - new_num_verts)
			<< " vertices in isolated regions." << std::endl;
	}

	/* Remove degenerated faces from the mesh. */
	if (conf.meshCleanSettings.clean_degenerated)
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
		ply_opts.write_vertex_colors = !conf.meshCleanSettings.delete_colors;
		ply_opts.write_vertex_confidences = !conf.meshCleanSettings.delete_conf;
		ply_opts.write_vertex_values = !conf.meshCleanSettings.delete_scale;
		mve::geom::save_ply_mesh(mesh, cleaned_mesh_file, ply_opts);
	}
	else
	{
		mve::geom::save_mesh(mesh, cleaned_mesh_file);
	}

	log_message(conf, "FSSR Mesh Cleaning ends.");

	return EXIT_SUCCESS;
}