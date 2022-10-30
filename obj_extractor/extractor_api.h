#pragma once
#include <vector>
#ifdef OBJ_EXTRACT_EXPORTS
#define OBJ_EXTRACT_API __declspec(dllexport) 
#else
#define OBJ_EXTRACT_API __declspec(dllimport) 
#endif

OBJ_EXTRACT_API void extractObj(std::vector<std::vector<float>>& vertices, float plane_tolerance, float cluster_tolerance);
OBJ_EXTRACT_API void extractObj(std::string& inputPlyFileName, std::string& outPlyFileName, std::vector<float>& cam_positions,
								float plane_tolerance, float cluster_tolerance);
OBJ_EXTRACT_API void segmentPlane(std::vector<float>& points, float plane_tolerance, std::vector<int>& labels);
OBJ_EXTRACT_API void initCamParams();
OBJ_EXTRACT_API void createViewer(int x, int y, int width, int height, std::string &caption);
OBJ_EXTRACT_API void addImgPath(std::string& img_path);
OBJ_EXTRACT_API void getCamParams(float* position, float* rot, float* worldToCam, float* calib, float* inverse_calib, float focalLen);
OBJ_EXTRACT_API void loadMesh(std::string& path);
OBJ_EXTRACT_API void drawScene(int nType);
OBJ_EXTRACT_API void setting(float len, std::string& raw_file_name, std::string& decimate_file_name);
OBJ_EXTRACT_API void scalingMesh(float s);
OBJ_EXTRACT_API void decimatingMesh();
OBJ_EXTRACT_API float getScale();
OBJ_EXTRACT_API int getVertCount();
OBJ_EXTRACT_API void getVertices(float* vertices, float* normals, int vert_count);
