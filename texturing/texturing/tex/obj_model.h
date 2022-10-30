/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_OBJMODEL_HEADER
#define TEX_OBJMODEL_HEADER

#include "material_lib.h"
/**
  * Class representing a obj model.
  */
class ObjModel {
public:
    struct Face {
        std::size_t vertex_ids[3];
        std::size_t texcoord_ids[3];
        std::size_t normal_ids[3];
    };

    struct Group {
        std::string material_name;
        std::vector<Face> faces;
    };

	struct FaceInfo {
		int groupId;
		int faceId;
	};//added at 02/13/2019

	struct Edgetag {
		int vertex_ids[2];
		int texcoord_ids[3];
		int normal_ids[2];
		int groupId;
		int face_id;
	};//added at 02/14/2019

	struct Triangle {
		std::size_t vertex_ids[3];
		std::size_t texcoord_ids[3];
		std::size_t normal_ids[3];
		int groupId;
	};
    typedef std::vector<math::Vec3f> Vertices;
    typedef std::vector<math::Vec2f> TexCoords;
    typedef std::vector<math::Vec3f> Normals;
    typedef std::vector<Group> Groups;
	typedef std::vector<FaceInfo> FaceInfoes;//added at 02/13/2019
	typedef Edgetag Edge;
	typedef std::vector<Triangle> Triangles;
private:
    Vertices vertices;
    TexCoords texcoords;
    Normals normals;
    Groups groups;
    MaterialLib material_lib;

	int nBoundaries;
	int countOfNewFaces;
	int countOfEdges;
	
	void filteringGroup(Group &group, int nthreshold);//added at 02/13/2019
	void closeHoles(Group &group);//added at 02/14/2019
	void getTotalBoundaries(Triangles &triangles, std::vector<Edge> &boundary);
	bool isHole(std::vector<Edge> &edges, float diameter);
	void removingNonManifoldFaces(std::vector<Face> &faces);
	void removingPieces(int nThreshold);
	void refiningBoundary(std::vector<Edge> &boundary, Triangles &triangles, std::vector<Triangle> &newTries);
	void getTriangles(Triangles &triangles);
	void integrateMesh(Triangles triangles);
	void getCloseBoundaries(std::vector<Edge> &totalBoundary, std::vector<std::vector<Edge>> &boundaries);
	void removeErrorFaces(Triangles &triangles);
	void getHoleFillSamples(std::vector<Edge> &boundary, std::vector<math::Vec3f> &newVertices);
public:
    /** Saves the obj model to an .obj file, its material lib and the materials with the given prefix. */
    void save_to_files(std::string const & prefix) const;
	/*remove small pieces isolated*/
	void filtering(int nThreshold);
    MaterialLib & get_material_lib(void);
    Vertices & get_vertices(void);
    TexCoords & get_texcoords(void);
    Normals & get_normals(void);
    Groups & get_groups(void);
	int getCountBoundaries();
	int getCountOfNewFaces();
	int getCountOfGroups();
	int getcountOfEdges();
	void fillingHoles();
    static void save(ObjModel const & model, std::string const & prefix);
};

inline
MaterialLib &
ObjModel::get_material_lib(void) {
    return material_lib;
}

inline
ObjModel::Vertices &
ObjModel::get_vertices(void) {
    return vertices;
}

inline
ObjModel::TexCoords &
ObjModel::get_texcoords(void) {
    return texcoords;
}

inline
ObjModel::Normals &
ObjModel::get_normals(void) {
    return normals;
}

inline
ObjModel::Groups &
ObjModel::get_groups(void) {
    return groups;
}

#endif /* TEX_OBJMODEL_HEADER */
