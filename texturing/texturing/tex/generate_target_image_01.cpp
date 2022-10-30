#include "texturing.h"
TEX_NAMESPACE_BEGIN
/*
created at 08/02/2017
*/
void generate_face_project_info(mve::TriangleMesh::ConstPtr mesh, tex::TextureViews &texture_views, Face_Project_Infos &face_project_infos)
{
	std::vector<unsigned int> const & faces = mesh->get_faces();
	std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
	mve::TriangleMesh::NormalList const & face_normals = mesh->get_face_normals();
	face_project_infos.resize(faces.size()/3);
	std::size_t const num_views = texture_views.size();
	BVHTree bvh_tree(faces, vertices);
#pragma omp parallel
	{
#pragma omp for schedule(dynamic)
#if !defined(_MSC_VER)
		for (std::uint16_t i = 0; i < static_cast<std::uint16_t>(faces.size()); i += 3) {
#else
		for (std::int32_t i = 0; i < faces.size(); i += 3) {
#endif
			std::size_t face_id = i / 3;
			math::Vec3f const & v1 = vertices[faces[i]];
			math::Vec3f const & v2 = vertices[faces[i + 1]];
			math::Vec3f const & v3 = vertices[faces[i + 2]];
			math::Vec3f const & face_normal = face_normals[face_id];
			math::Vec3f const face_center = (v1 + v2 + v3) / 3.0f;
			face_project_infos[face_id].normal = face_normal;
			for (int j = 0; j < num_views; j++) {
				math::Vec3f const & view_pos = texture_views[j].get_pos();
				math::Vec3f const & viewing_direction = texture_views[j].get_viewing_direction();
				/* Check visibility and compute quality */
				math::Vec3f view_to_face_vec = (face_center - view_pos).normalized();
				math::Vec3f face_to_view_vec = (view_pos - face_center).normalized();
				/* Backface and basic frustum culling */
				float viewing_angle = face_to_view_vec.dot(face_normal);
				if (viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)
					continue;

				if (std::acos(viewing_angle) > MATH_DEG2RAD(75.0f))
					continue;
				/* Projects into the valid part of the TextureView? */
				if (!texture_views[j].inside(v1, v2, v3))
					continue;
				/* Viewing rays do not collide? */
				bool visible = true;
				math::Vec3f const * samples[] = { &v1, &v2, &v3 };
				// TODO: random monte carlo samples...
				for (std::size_t k = 0; k < sizeof(samples) / sizeof(samples[0]); ++k) {
					BVHTree::Ray ray;
					ray.origin = *samples[k];
					ray.dir = view_pos - ray.origin;
					ray.tmax = ray.dir.norm();
					ray.tmin = ray.tmax * 0.0001f;
					ray.dir.normalize();
					BVHTree::Hit hit;
					if (bvh_tree.intersect(ray, &hit)) {
						visible = false;
						break;
					}
				}
				if (!visible) continue;
				face_project_infos[face_id].view_ids.push_back(j);
			}
		}
	}
}

/*
created at 07/24/2017
*/
void generate_target_images(mve::TriangleMesh::ConstPtr mesh, tex::TextureViews &texture_views, 
	int nIterations, int patch_width, int patch_height)
{
	int loop = nIterations;
	int loop_count = 0;
	while (loop > 0)
	{
		std::cout << "start iterations:" << loop_count << std::endl;
		for (int i = 0; i < texture_views.size(); i++)
		{
			texture_views[i].update_target_image(texture_views);
		}
		for (int i = 0; i < texture_views.size(); i++)
		{
			//std::cout << "start texture map:" << i << std::endl;
			texture_views[i].update_texture_map(texture_views);
			//std::cout << "pass texture map:" << i << std::endl;
		}
		std::cout << "iterations:" << loop_count << std::endl;
		loop--;
		loop_count++;
	}
}
/*
created at 08/09/2017
*/
void initialize(mve::TriangleMesh::ConstPtr mesh, tex::TextureViews &texture_views,
				int patch_width, int patch_height, Face_Project_Infos &face_project_infos)
{
	generate_face_project_info(mesh, texture_views, face_project_infos);
	int view_count = texture_views.size();
	for (int i = 0; i < view_count; i++)
	{
		texture_views[i].initialize(patch_width, patch_height);
		//texture_views[i].generate_pixel_maps(mesh, texture_views);
		//texture_views[i].set_face_project_info(&face_project_infos);
		//texture_views[i].generate_samples(mesh);
	}
	for (int i = 0; i < view_count; i++)
	{
		texture_views[i].generate_pixel_maps(mesh, texture_views);
	}
	//std::cout << "compelete all pixel maps." << std::endl;
}
/*
created at 08/08/2017
*/
void down_sampling(tex::TextureViews &texture_views, int down_res)
{
	for (int i = 0; i < texture_views.size(); i++)
	{
		texture_views.at(i).down_sampling(down_res);
	}
}
/*
created at 08/08/2017
*/
void up_sampling(tex::TextureViews &texture_views, int up_res)
{
	for (int i = 0; i < texture_views.size(); i++)
	{
		texture_views.at(i).up_sampling(up_res);
	}
}
void texture_view_reset(tex::TextureViews &texture_views)
{
	for (int i = 0; i < texture_views.size(); i++)
	{
		texture_views[i].reset_projection();
	}
}
/*
created at 08/08/2017
*/
void multi_scale_optimization(mve::TriangleMesh::ConstPtr mesh, tex::TextureViews &texture_views,
							  int nIterations, int patch_size, int min_res)
{
	float scale = (float)texture_views[0].get_height() / min_res;
	float y = (float)1 / 9;
	scale = std::powf(scale, y);
	Face_Project_Infos face_project_infos;
	initialize(mesh, texture_views, patch_size, patch_size, face_project_infos);
	//down_sampling(texture_views, min_res);
	generate_target_images(mesh, texture_views, nIterations, patch_size, patch_size);
	/*std::cout << "pass generate downsamped target" << std::endl;
	int up_res = min_res;
	for (int i = 1; i < 10; i++)
	{
		std::cout << "start upsampling:" << i << std::endl;
		up_res = min_res*std::powf(scale, i);
		//std::cout << "start upsampling:" << i << std::endl;
		up_sampling(texture_views, up_res);
		//std::cout << "pass upsampling:" << i << std::endl;
		generate_target_images(mesh, texture_views, 5, patch_size, patch_size);
		//std::cout << "generate rsolution:" << up_res << std::endl;
	}*/
	for (int i = 0; i < texture_views.size(); i++)
	{
		texture_views[i].save_target_image();
		//texture_views[i].save_texture_map();
	}
	//texture_view_reset(texture_views);
}
TEX_NAMESPACE_END