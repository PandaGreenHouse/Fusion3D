#pragma once
#include <iostream>
#ifdef RECON_EXPORTS
#define RECON_API __declspec(dllexport) 
#else
#define RECON_API __declspec(dllimport) 
#endif

typedef void(*On3DProgressCallback)(int total_progress, int sub_progress, char* msg);

RECON_API void init_recon(std::string& img_set_path, std::string& scene_path, int match_mode, int mvsType,
						  int max_pixels_level, int dm_input_scale, int dm_out_scale, bool use_shading,
						  float plane_tolerance, float cluster_tolerance, On3DProgressCallback pProgressCallback);
RECON_API void make_scene();
RECON_API void sfm_recon();
RECON_API void dm_recon(int mvsType);
RECON_API void fss_recon();
RECON_API void tex_recon(std::string& scene_path, float s);
RECON_API void recon_all(std::string& img_set_path, std::string& scene_path, int match_mode, int mvsType);
RECON_API int loadScene(std::string &path);
RECON_API void scalingModel(float real_focal_len);
