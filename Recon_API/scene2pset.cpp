#include "common.h"

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

void
poisson_scale_normals(mve::TriangleMesh::ConfidenceList const& confs,
	mve::TriangleMesh::NormalList* normals)
{
	if (confs.size() != normals->size())
		throw std::invalid_argument("Invalid confidences or normals");
	for (std::size_t i = 0; i < confs.size(); ++i)
		normals->at(i) *= confs[i];
}

int scene2pset(AppSettings& conf)
{
	log_message(conf, "Scene to point set starts.");

	// 	util::system::register_segfault_handler();
	// 	util::system::print_build_timestamp("MVE Scene to Pointset");

	/* Init default settings. */
	conf.psetSettings.pset_name = util::fs::join_path(conf.sceneSettings.path_scene, "pset-L2.ply");
	conf.psetSettings.with_conf = true;
	conf.psetSettings.with_normals = true;
	conf.psetSettings.with_scale = true;
	int const scale = conf.dmSettings.mvs.scale;//2;
	conf.psetSettings.dmname = "depth-L" + util::string::get<int>(scale);
	conf.psetSettings.image = (scale == 0) ? "undistorted" : "undist-L" + util::string::get<int>(scale);

	if (util::string::right(conf.psetSettings.pset_name, 5) == ".npts" || util::string::right(conf.psetSettings.pset_name, 6) == ".bnpts")
	{
		conf.psetSettings.with_normals = true;
		conf.psetSettings.with_scale = false;
		conf.psetSettings.with_conf = false;
	}

	if (conf.psetSettings.poisson_normals)
	{
		conf.psetSettings.with_normals = true;
		conf.psetSettings.with_conf = true;
	}

	/* If requested, use given AABB. */
	math::Vec3f aabbmin, aabbmax;
	if (!conf.psetSettings.aabb.empty()) {
		aabb_from_string(conf.psetSettings.aabb, &aabbmin, &aabbmax);
	}

	// 	std::cout << "Using depthmap \"" << conf.dmname << "\" and color image \"" << conf.image << "\"" << std::endl;
	log_message(conf, "Using depthmap \"" + util::string::get(conf.psetSettings.dmname) + "\" and color image \"" + util::string::get(conf.psetSettings.image) + "\"");

	/* Prepare output mesh. */
	mve::TriangleMesh::Ptr pset(mve::TriangleMesh::create());
	mve::TriangleMesh::VertexList& verts(pset->get_vertices());
	mve::TriangleMesh::NormalList& vnorm(pset->get_vertex_normals());
	mve::TriangleMesh::ColorList& vcolor(pset->get_vertex_colors());
	mve::TriangleMesh::ValueList& vvalues(pset->get_vertex_values());
	mve::TriangleMesh::ConfidenceList& vconfs(pset->get_vertex_confidences());

	/* Load scene. */
	mve::Scene::Ptr scene = mve::Scene::create(conf.sceneSettings.path_scene);

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
		if (!conf.psetSettings.ids.empty() && std::find(conf.psetSettings.ids.begin(), conf.psetSettings.ids.end(), view_id) == conf.psetSettings.ids.end()) {
			continue;
		}

		mve::CameraInfo const& cam = view->get_camera();
		if (cam.flen == 0.0f) {
			continue;
		}

		mve::FloatImage::Ptr dm = view->get_float_image(conf.psetSettings.dmname);
		if (dm == nullptr) {
			continue;
		}

		if (conf.psetSettings.min_valid_fraction > 0.0f)
		{
			float num_total = static_cast<float>(dm->get_value_amount());
			float num_recon = 0;
			for (int j = 0; j < dm->get_value_amount(); ++j) {
				if (dm->at(j) > 0.0f) {
					num_recon += 1.0f;
				}
			}

			float fraction = num_recon / num_total;
			if (fraction < conf.psetSettings.min_valid_fraction)
			{
				// 				std::cout << "View " << view->get_name() << ": Fill status " << util::string::get_fixed(fraction * 100.0f, 2) << "%, skipping." << std::endl;
				log_message(conf, "View " + util::string::get(view->get_name()) + ": Fill status " + util::string::get(util::string::get_fixed(fraction * 100.0f, 2)) + "%, skipping.");
				continue;
			}
		}

		mve::ByteImage::Ptr ci;
		if (!conf.psetSettings.image.empty()) {
			ci = view->get_byte_image(conf.psetSettings.image);
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

		if (conf.psetSettings.with_normals) {
			mesh->ensure_normals();
		}

		/* If confidence is requested, compute it. */
		if (conf.psetSettings.with_conf)
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

		if (conf.psetSettings.poisson_normals) {
			poisson_scale_normals(mconfs, &mesh->get_vertex_normals());
		}

		/* If scale is requested, compute it. */
		std::vector<float> mvscale;
		if (conf.psetSettings.with_scale)
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
				mvscale[j] *= conf.psetSettings.scale_factor;
			}
		}

		/* Add vertices and optional colors and normals to mesh. */
		if (conf.psetSettings.aabb.empty())
		{
			/* Fast if no bounding box is given. */
#pragma omp critical
		{
			verts.insert(verts.end(), mverts.begin(), mverts.end());
			if (!mvcol.empty())
				vcolor.insert(vcolor.end(), mvcol.begin(), mvcol.end());
			if (conf.psetSettings.with_normals)
				vnorm.insert(vnorm.end(), mnorms.begin(), mnorms.end());
			if (conf.psetSettings.with_scale)
				vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());
			if (conf.psetSettings.with_conf)
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
					if (conf.psetSettings.with_normals)
						vnorm.push_back(mnorms[i]);
					if (conf.psetSettings.with_scale)
						vvalues.push_back(mvscale[i]);
					if (conf.psetSettings.with_conf)
						vconfs.push_back(mconfs[i]);
				}
			}
		}

		dm.reset();
		ci.reset();
		view->cache_cleanup();
	}

	/* If a mask is given, clip vertices with the masks in all images. */
	if (!conf.psetSettings.mask.empty())
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

			mve::ByteImage::Ptr mask = view->get_byte_image(conf.psetSettings.mask);
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
	if (util::string::right(conf.psetSettings.pset_name, 4) == ".ply")
	{
		mve::geom::SavePLYOptions opts;
		opts.write_vertex_normals = conf.psetSettings.with_normals;
		opts.write_vertex_values = conf.psetSettings.with_scale;
		opts.write_vertex_confidences = conf.psetSettings.with_conf;
		mve::geom::save_ply_mesh(pset, conf.psetSettings.pset_name, opts);
	}
	else
	{
		mve::geom::save_mesh(pset, conf.psetSettings.pset_name);
	}
	/*added at 2022/08/03, for extracting object*/
	mve::TriangleMesh::Ptr pset_obj(mve::TriangleMesh::create());
	extractObj(scene, conf.plane_tolerance, pset, pset_obj);
	if (util::string::right(conf.psetSettings.pset_name1, 4) == ".ply")
	{
		mve::geom::SavePLYOptions opts;
		opts.write_vertex_normals = conf.psetSettings.with_normals;
		opts.write_vertex_values = conf.psetSettings.with_scale;
		opts.write_vertex_confidences = conf.psetSettings.with_conf;
		mve::geom::save_ply_mesh(pset_obj, conf.psetSettings.pset_name1, opts);
	}
	else
	{
		mve::geom::save_mesh(pset_obj, conf.psetSettings.pset_name1);
	}
	log_message(conf, "object is extract and saved.");
	log_message(conf, "Scene to point set ends.");

	return EXIT_SUCCESS;
}
