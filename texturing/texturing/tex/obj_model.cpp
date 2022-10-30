/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iomanip>
#include <fstream>
#include <cstring>
#include <cerrno>

#include <mve/mesh.h>
#include <util/exception.h>
#include <util/file_system.h>

#include "obj_model.h"

#define OBJ_INDEX_OFFSET 1
#define RAD1 1.308996938996
#define RAD2 2.3561944901923
#define PAI  3.1415926535898
void
ObjModel::save(ObjModel const & model, std::string const & prefix) {

	model.save_to_files(prefix);
}

void
ObjModel::save_to_files(std::string const & prefix) const {
	material_lib.save_to_files(prefix);
	std::string name = util::fs::basename(prefix);
    std::ofstream out((prefix + ".obj").c_str());
    if (!out.good())
		throw util::FileException(prefix + ".obj", "ObjModel::save_to_files");// std::strerror(errno));

    out << "mtllib " << name << ".mtl" << std::endl;

    out << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        out << "v " << vertices[i][0] << " "
            << vertices[i][1] << " "
            << vertices[i][2] << std::endl;
    }

    for (std::size_t i = 0; i < texcoords.size(); ++i) {
        out << "vt " << texcoords[i][0] << " "
            << 1.0f - texcoords[i][1] << std::endl;
    }

    for (std::size_t i = 0; i < normals.size(); ++i) {
        out << "vn " << normals[i][0] << " "
            << normals[i][1] << " "
            << normals[i][2] << std::endl;
    }

    for (std::size_t i = 0; i < groups.size(); ++i) {
        out << "usemtl " << groups[i].material_name << std::endl;
        for (std::size_t j = 0; j < groups[i].faces.size(); ++j) {
            Face const & face =  groups[i].faces[j];
            out << "f";
            for (std::size_t k = 0; k < 3; ++k) {
                out << " " << face.vertex_ids[k]  + OBJ_INDEX_OFFSET
                    << "/" << face.texcoord_ids[k]  + OBJ_INDEX_OFFSET
                    << "/" << face.normal_ids[k]  + OBJ_INDEX_OFFSET;
            }
            out << std::endl;
        }
    }
    out.close();
}
/*
created at 02/19/2019
*/
void ObjModel::removingNonManifoldFaces(std::vector<Face> &faces)
{
	std::vector<int> nonManifolds;
	for (int i = 0; i < faces.size(); ++i)
	{
		Edge edges[3];
		int counts[3];
		for (int j = 0; j < 3; ++j)
		{
			edges[j].vertex_ids[0] = faces[i].vertex_ids[j];
			edges[j].vertex_ids[1] = faces[i].vertex_ids[(j+1)%3];
			counts[j] = 0;
		}
		for (int j = 0; j < faces.size(); ++j)
		{
			if (i == j) continue;
			for (int k = 0; k < 3; ++k)
			{
				int count = 0;
				for (int n = 0; n < 3; ++n)
				{
					if (edges[k].vertex_ids[0] == faces[j].vertex_ids[n])
						++count;
					if (edges[k].vertex_ids[1] == faces[j].vertex_ids[n])
						++count;
				}
				if (count == 2)
					++counts[k];
			}
		}
		int count1 = 0;
		int count2 = 0;
		for (int j = 0; j < 3; ++j)
		{
			if (counts[j] == 2)
				count1 = 1;
			if (counts[j] == 1)
				count2 = 1;
		}
		if (count1 == 1 && count2 == 0)
			nonManifolds.push_back(i);
	}
	for (int i = nonManifolds.size() - 1; i > -1; --i)
	{
		faces.erase(faces.begin() + i);
	}
}
/*
created at 02/13/2019
remove small isolated pieces, but it may remove adjacent groups
*/
void ObjModel::filteringGroup(Group &group, int nThreshold)
{
	std::vector<Face> faces = group.faces;
	int nFaces = faces.size();
	std::vector<int> fLabels;
	fLabels.resize(nFaces);
	for (int i = 0; i < nFaces; ++i)
	{
		fLabels[i] = 1;
	}
	std::vector<std::vector<int>> pieces;
	for (int i = 0; i < nFaces; ++i)
	{
		if (fLabels[i] == 0) continue;
		fLabels[i] = 0;
		std::vector<int> tmp, piece;
		tmp.push_back(i);
		piece.push_back(i);
		while (tmp.size() > 0)
		{
			int t = tmp.at(0);
			tmp.erase(tmp.begin());
			Face f1 = faces[t];
			for (int j = 0; j < nFaces; ++j)
			{
				if (fLabels[j] == 0) continue;
				if (t == j) continue;
				Face f2 = faces[j];
				bool flg = false;
				for (int k = 0; k < 3; ++k)
				{
					if (f1.vertex_ids[0] == f2.vertex_ids[k])
					{
						flg = true;
						break;
					}
					if (f1.vertex_ids[1] == f2.vertex_ids[k])
					{
						flg = true;
						break;
					}
					if (f1.vertex_ids[2] == f2.vertex_ids[k])
					{
						flg = true;
						break;
					}
				}
				if (flg == true)
				{
					tmp.push_back(j);
					piece.push_back(j);
					fLabels[j] = 0;
				}
			}
		}
		if (piece.size() > 0)
			pieces.push_back(piece);
	}
	std::vector<Face> faces1;
	for (int i = 0; i < pieces.size(); ++i)
	{
		if (pieces[i].size() < nThreshold) continue;
		for (int j = 0; j < pieces[i].size(); ++j)
		{
			Face f = faces[pieces[i][j]];
			faces1.push_back(f);
		}
	}
	group.faces.swap(faces1);
}
/*
created at 02/13/2019
modified at 02/19/2019
*/
void ObjModel::filtering(int nThreshold)
{
	nBoundaries = 0;
	countOfNewFaces = 0;
	countOfEdges = 0;
	if (nThreshold > 0)
	{
		removingPieces(nThreshold);
		/*for (int i = 0; i < groups.size(); ++i)
		{
			filteringGroup(groups[i], nThreshold);
		}*/
		fillingHoles();
		/*for (int i = 0; i < groups.size(); ++i)
		{
			removingNonManifoldFaces(groups[i].faces);
		}*/
	}
}
/*
created at 02/14/2019
*/
void ObjModel::closeHoles(Group &group)
{
	int nFaces = group.faces.size();
	std::vector<Edge> totalBoundary;
	for (int i = 0; i < nFaces; ++i)
	{
		Edge edges[3];
		for (int j = 0; j < 3; ++j)
		{
			edges[j].vertex_ids[0] = group.faces[i].vertex_ids[j];
			edges[j].vertex_ids[1] = group.faces[i].vertex_ids[(j + 1) % 3];
			edges[j].texcoord_ids[0] = group.faces[i].texcoord_ids[j];
			edges[j].texcoord_ids[1] = group.faces[i].texcoord_ids[(j + 1) % 3];
			edges[j].normal_ids[0] = group.faces[i].normal_ids[j];
			edges[j].normal_ids[1] = group.faces[i].normal_ids[(j+1)%3];
		}
		for (int j = 0; j < 3; ++j)
		{
			int count = 0;
			for (int k = 0; k < nFaces; ++k)
			{
				count = 0;
				if (k == i) continue;
				for (int n = 0; n < 3; ++n)
				{
					if (edges[j].vertex_ids[0] == group.faces[k].vertex_ids[n] || edges[j].vertex_ids[1] == group.faces[k].vertex_ids[n])
					{
						++count;
					}
				}
				if (count == 2)
				{
					break;
				}
			}
			if (count < 2)
				totalBoundary.push_back(edges[j]);
		}
	}
	countOfEdges = totalBoundary.size();
	std::vector<std::vector<Edge>> boundaries;
	int count0 = 0;
	while (totalBoundary.size() > 1)
	{
		std::vector<Edge> boundary;
		Edge e = totalBoundary.at(0);
		boundary.push_back(e);
		totalBoundary.erase(totalBoundary.begin());
		int count = 1;
		while (count > 0)
		{
			count = 0;
			for (int i = 0; i < totalBoundary.size(); ++i)
			{
				Edge e2 = totalBoundary.at(i);
				if (e.vertex_ids[0] == e2.vertex_ids[0] || e.vertex_ids[0] == e2.vertex_ids[1])
				{
					e = e2;
					boundary.push_back(e2);
					totalBoundary.erase(totalBoundary.begin() + i);
					count = 1;
					break;
				}
				if (e.vertex_ids[1] == e2.vertex_ids[0] || e.vertex_ids[1] == e2.vertex_ids[1])
				{
					e = e2;
					boundary.push_back(e2);
					totalBoundary.erase(totalBoundary.begin() + i);
					count = 1;
					break;
				}
			}
		}
		if (boundary.size() > 2)
		{
			Edge e0 = boundary[0];
			int nCount = 0;
			for (int i = 1; i < boundary.size(); ++i)
			{
				if (e0.vertex_ids[0] == boundary[i].vertex_ids[0])
					++nCount;
				if (e0.vertex_ids[1] == boundary[i].vertex_ids[0])
					++nCount;
				if (e0.vertex_ids[0] == boundary[i].vertex_ids[1])
					++nCount;
				if (e0.vertex_ids[1] == boundary[i].vertex_ids[1])
					++nCount;
				if (nCount == 2) break;
			}
			if(nCount==2)
				boundaries.push_back(boundary);
		}
			
		//if (boundary.size() > 2)
		//{
			/*++count0;
			Edge e0 = boundary.at(0);
			int nCount = 0;
			for (int i = 1; i < boundary.size(); ++i)
			{
				if (e0.v1 == boundary.at(i).v1)
					++nCount;
				if (e0.v1 == boundary.at(i).v2)
					++nCount;
				if (e0.v2 == boundary.at(i).v1)
					++nCount;
				if (e0.v2 == boundary.at(i).v2)
					++nCount;
				if (nCount == 2)
					break;
			}
			if (nCount == 2)
				boundaries.push_back(boundary);*/
			//boundaries.push_back(boundary);
		//}
	}
	nBoundaries = boundaries.size();
	countOfNewFaces = 0;
	for (int i = 0; i < boundaries.size(); ++i)
	{
		math::Vec3f vCenter(0.0f, 0.0f, 0.0f);
		math::Vec3f vNormal(0.0f, 0.0f, 0.0f);
		math::Vec2f vTexCoord(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < boundaries[i].size(); ++j)
		{
			vCenter += vertices[boundaries[i][j].vertex_ids[0]];
			vNormal += normals[boundaries[i][j].normal_ids[0]];
			vTexCoord += texcoords[boundaries[i][j].texcoord_ids[0]];
			
			Face f;
			f.vertex_ids[0] = boundaries[i][j].vertex_ids[0];
			f.vertex_ids[1] = vertices.size();
			f.vertex_ids[2] = boundaries[i][j].vertex_ids[1];
						
			f.texcoord_ids[0] = boundaries[i][j].texcoord_ids[0];
			f.texcoord_ids[1] = texcoords.size();
			f.texcoord_ids[2] = boundaries[i][j].texcoord_ids[1];
									
			f.normal_ids[0] = boundaries[i][j].normal_ids[0];
			f.normal_ids[1] = normals.size();
			f.normal_ids[2] = boundaries[i][j].normal_ids[1];
									
			group.faces.push_back(f);
			++countOfNewFaces;
		}
		vCenter /= boundaries[i].size();
		vNormal /= boundaries[i].size();
		vTexCoord /= boundaries[i].size();
		vertices.push_back(vCenter);
		normals.push_back(vNormal);
		texcoords.push_back(vTexCoord);
	}
}

int ObjModel::getCountBoundaries()
{
	return nBoundaries;
}

int ObjModel::getCountOfNewFaces()
{
	return countOfNewFaces;
}

int ObjModel::getCountOfGroups()
{
	return groups.size();
}
int ObjModel::getcountOfEdges()
{
	return countOfEdges;
}
/*
created at 02/25/2019
*/
void ObjModel::getTriangles(Triangles &triangles)
{
	for (int i = 0; i < groups.size(); ++i)
	{
		for (int j = 0; j < groups[i].faces.size(); ++j)
		{
			Triangle t;
			Face const & f = groups[i].faces[j];
			for (int k = 0; k < 3; ++k)
			{
				t.vertex_ids[k] = f.vertex_ids[k];
				t.texcoord_ids[k] = f.texcoord_ids[k];
				t.normal_ids[k] = f.normal_ids[k];
			}
			t.groupId = i;
			triangles.push_back(t);
		}
	}
}
/*
created at 02/18/2019
modified at 02/21/2019
*/
void ObjModel::getTotalBoundaries(Triangles &triangles, std::vector<Edge> &boundary)
{
	for (int i = 0; i < triangles.size(); ++i)
	{
		Triangle const & t = triangles[i];
		Edge edges[3];
		for (int j = 0; j < 3; ++j)
		{
			edges[j].vertex_ids[0] = t.vertex_ids[j];
			edges[j].vertex_ids[1] = t.vertex_ids[(j+1)%3];
			edges[j].texcoord_ids[0] = t.texcoord_ids[j];
			edges[j].texcoord_ids[1] = t.texcoord_ids[(j+1)%3];
			edges[j].texcoord_ids[2] = t.texcoord_ids[(j + 2) % 3];
			edges[j].normal_ids[0] = t.normal_ids[j];
			edges[j].normal_ids[1] = t.normal_ids[(j + 1) % 3];
			edges[j].groupId = t.groupId;
			edges[j].face_id = i;
		}
		for (int j = 0; j < 3; ++j)
		{
			int count = 0;
			for (int k = 0; k < triangles.size(); ++k)
			{
				if (i == k) continue;
				Triangle const & t1 = triangles[k];
				count = 0;
				for (int n = 0; n < 3; ++n)
				{
					if (edges[j].vertex_ids[0] == t1.vertex_ids[n])
						++count;
					if (edges[j].vertex_ids[1] == t1.vertex_ids[n])
						++count;
				}
				if (count == 2) break;
			}
			if (count < 2)
				boundary.push_back(edges[j]);
		}
	}
}
/*
created at 02/25/2019
*/
void ObjModel::integrateMesh(Triangles newTriangles)
{
	for (int i = 0; i < newTriangles.size(); ++i)
	{
		Triangle t = newTriangles[i];
		Face f;
		for (int j = 0; j < 3; ++j)
		{
			f.vertex_ids[j] = t.vertex_ids[j];
			f.normal_ids[j] = t.normal_ids[j];
			f.texcoord_ids[j] = t.texcoord_ids[j];
		}
		groups[t.groupId].faces.push_back(f);
	}
}
/*
created at 02/25/2019
*/
void ObjModel::getCloseBoundaries(std::vector<Edge> &total_boundary, std::vector<std::vector<Edge>> &boundaries)
{
	while (total_boundary.size() > 1)
	{
		Edge edge = total_boundary[0];
		total_boundary.erase(total_boundary.begin());
		std::vector<Edge> tmp, boundary;
		tmp.push_back(edge);
		boundary.push_back(edge);
		while (tmp.size() > 0)
		{
			Edge e = tmp[0];
			tmp.erase(tmp.begin());
			for (int j = 0; j < total_boundary.size(); ++j)
			{
				if (e.vertex_ids[1] == total_boundary[j].vertex_ids[0])
				{
					Edge e1;
					e1.vertex_ids[0] = total_boundary[j].vertex_ids[0];
					e1.vertex_ids[1] = total_boundary[j].vertex_ids[1];
					e1.groupId = total_boundary[j].groupId;
					e1.face_id = total_boundary[j].face_id;
					e1.normal_ids[0] = total_boundary[j].normal_ids[0];
					e1.normal_ids[1] = total_boundary[j].normal_ids[1];
					e1.texcoord_ids[0] = total_boundary[j].texcoord_ids[0];
					e1.texcoord_ids[1] = total_boundary[j].texcoord_ids[1];
					e1.texcoord_ids[2] = total_boundary[j].texcoord_ids[2];
					tmp.push_back(e1);
					boundary.push_back(e1);
					total_boundary.erase(total_boundary.begin() + j);
					break;
				}
				if (e.vertex_ids[1] == total_boundary[j].vertex_ids[1])
				{
					Edge e1;
					e1.vertex_ids[0] = total_boundary[j].vertex_ids[1];
					e1.vertex_ids[1] = total_boundary[j].vertex_ids[0];
					e1.groupId = total_boundary[j].groupId;
					e1.face_id = total_boundary[j].face_id;
					e1.normal_ids[0] = total_boundary[j].normal_ids[1];
					e1.normal_ids[1] = total_boundary[j].normal_ids[0];
					e1.texcoord_ids[0] = total_boundary[j].texcoord_ids[1];
					e1.texcoord_ids[1] = total_boundary[j].texcoord_ids[0];
					e1.texcoord_ids[2] = total_boundary[j].texcoord_ids[2];
					tmp.push_back(e1);
					boundary.push_back(e1);
					total_boundary.erase(total_boundary.begin() + j);
					break;
				}
			}
		}
		if (isHole(boundary, 0.6) == false)
			continue;
		if (boundary.size()>2)
			boundaries.push_back(boundary);
	}
}
/*
created at 02/26/2019
*/
void ObjModel::removeErrorFaces(Triangles &triangles)
{
	int nPos = 0;
	while( nPos < triangles.size())
	{
		bool flg = false;
		for (int j = 0; j < 3; ++j)
		{
			for (int n = j+1; n < j+3; ++n)
			{
				if (triangles[nPos].vertex_ids[j] == triangles[nPos].vertex_ids[n % 3])
				{
					flg = true;
					break;
				}
			}
			if (flg == true)
				break;
		}
		if (flg == true)
		{
			triangles.erase(triangles.begin() + nPos);
			continue;
		}
		++nPos;
	}
}
/*
crerated at 02/18/2019
modified at 02/20/2019
modified at 02/25/2019
modified at 02/26/2019
*/
void ObjModel::fillingHoles()
{
	std::vector<Edge> total_boundary;
	std::vector<std::vector<Edge>> boundaries;
	Triangles triangles;
	getTriangles(triangles);
	getTotalBoundaries(triangles, total_boundary);
	getCloseBoundaries(total_boundary, boundaries);
	int nLoop = 5;
	Triangles newTriangles;
	while (nLoop > 0)
	{
		for (int i = 0; i < boundaries.size(); ++i)
		{
			refiningBoundary(boundaries[i], triangles, newTriangles);
		}
		--nLoop;
	}
	removeErrorFaces(newTriangles);
	integrateMesh(newTriangles);
	return;
	for (int i = 0; i < boundaries.size(); ++i)
	{
		math::Vec3f vCenter(0.0f, 0.0f, 0.0f);
		math::Vec3f vNormal(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < boundaries[i].size(); ++j)
		{
			vCenter += vertices[boundaries[i][j].vertex_ids[0]];
			vNormal += vertices[boundaries[i][j].normal_ids[0]];
		}
		vCenter /= boundaries[i].size();
		vNormal /= boundaries[i].size();
		vertices.push_back(vCenter);
		normals.push_back(vNormal);
		for (int j = 0; j < boundaries[i].size(); ++j)
		{
			Face f;
			f.vertex_ids[0] = boundaries[i][j].vertex_ids[0];
			f.vertex_ids[1] = vertices.size() - 1;
			f.vertex_ids[2] = boundaries[i][j].vertex_ids[1];

			f.normal_ids[0] = boundaries[i][j].normal_ids[0];
			f.normal_ids[1] = normals.size() - 1;
			f.normal_ids[2] = boundaries[i][j].normal_ids[1];

			f.texcoord_ids[0] = boundaries[i][j].texcoord_ids[0];
			f.texcoord_ids[1] = boundaries[i][j].texcoord_ids[1];
			f.texcoord_ids[2] = boundaries[i][j].texcoord_ids[2];

			int groupId = boundaries[i][j].groupId;
			groups[groupId].faces.push_back(f);
		}
	}
	/*for (int i = 0; i < boundaries.size(); ++i)
	{
		math::Vec3f vCenter(0.0f, 0.0f, 0.0f);
		math::Vec3f vNormal(0.0f, 0.0f, 0.0f);
		std::vector<std::vector<Edge>> group_edges;
		group_edges.resize(groups.size());
		for (int j = 0; j < boundaries[i].size(); ++j)
		{
			Edge e = boundaries[i][j];
			vCenter += vertices[e.vertex_ids[0]];
			vNormal += normals[e.normal_ids[0]];
			int groupId = boundaries[i][j].groupId;
			group_edges[groupId].push_back(boundaries[i][j]);
		}
		for (int j = 0; j < groups.size(); ++j)
		{
			if (group_edges[j].size() == 0)
				continue;
			math::Vec2f texCoord(0.0f, 0.0f);
			for (int k = 0; k < group_edges[j].size(); ++k)
			{
				texCoord += texcoords[group_edges[j][k].texcoord_ids[0]];
				Face f;
				f.vertex_ids[0] = group_edges[j][k].vertex_ids[0];
				f.vertex_ids[1] = vertices.size();
				f.vertex_ids[2] = group_edges[j][k].vertex_ids[1];
				f.normal_ids[0] = group_edges[j][k].normal_ids[0];
				f.normal_ids[1] = normals.size();
				f.normal_ids[2] = group_edges[j][k].normal_ids[1];
				f.texcoord_ids[0] = group_edges[j][k].texcoord_ids[0];
				f.texcoord_ids[1] = group_edges[j][k].texcoord_ids[1];//texcoords.size();
				f.texcoord_ids[2] = group_edges[j][k].texcoord_ids[2];
				groups[j].faces.push_back(f);
			}
			texCoord /= group_edges[j].size();
			texcoords.push_back(texCoord);
		}
		vCenter /= boundaries[i].size();
		vNormal /= boundaries[i].size();
		vertices.push_back(vCenter);
		normals.push_back(vNormal);
	}*/
}
/*
created at 02/19/2019
*/
bool ObjModel::isHole(std::vector<Edge> &edges, float diameter)
{
	float maxDist = 0.0f;
	std::vector<math::Vec3f> verts;
	for (int i = 0; i < edges.size(); ++i)
	{
		verts.push_back(vertices[edges[i].vertex_ids[0]]);
		verts.push_back(vertices[edges[i].vertex_ids[1]]);
	}
	for (int i = 0; i < verts.size(); ++i)
	{
		for (int j = i + 1; j < verts.size(); ++j)
		{
			math::Vec3f vDist = verts[j] - verts[i];
			if (vDist.norm() > maxDist)
			{
				maxDist = vDist.norm();
			}
			if (maxDist > diameter)
				return false;
		}
	}
	return true;
}
/*
created at 02/21/2019
*/
void ObjModel::removingPieces(int nThreshold)
{
	Triangles triangles;
	for (int i = 0; i < groups.size(); ++i)
	{
		std::vector<Face> const& faces = groups[i].faces;
		for (int j = 0; j < faces.size(); ++j)
		{
			Triangle t;
			for (int k = 0; k < 3; ++k)
			{
				t.vertex_ids[k] = faces[j].vertex_ids[k];
				t.texcoord_ids[k] = faces[j].texcoord_ids[k];
				t.normal_ids[k] = faces[j].normal_ids[k];
				t.groupId = i;
			}
			triangles.push_back(t);
		}
	}
	std::vector<int> labels;
	labels.resize(triangles.size());
	for (int i = 0; i < labels.size(); ++i)
	{
		labels[i] = 0;
	}
	std::vector<std::vector<int>> pieces;
	for (int i = 0; i < triangles.size(); ++i)
	{
		if (labels[i] == 1) continue;
		std::vector<int> tmp, piece;
		tmp.push_back(i);
		piece.push_back(i);
		while (tmp.size() > 0)
		{
			int triId = tmp[0];
			tmp.erase(tmp.begin());
			for (int j = 0; j < triangles.size(); ++j)
			{
				if (triId == j) continue;
				if (labels[j] == 1) continue;
				int count = 0;
				for (int k = 0; k < 3; ++k)
				{
					if (triangles[triId].vertex_ids[0] == triangles[j].vertex_ids[k])
							++count;
					if (triangles[triId].vertex_ids[1] == triangles[j].vertex_ids[k])
						++count;
					if (triangles[triId].vertex_ids[2] == triangles[j].vertex_ids[k])
						++count;
				}
				if (count > 1)
				{
					tmp.push_back(j);
					piece.push_back(j);
					labels[j] = 1;
				}
			}
		}
		if (piece.size() > nThreshold)
			pieces.push_back(piece);
	}
	for (int i = 0; i < groups.size(); ++i)
	{
		groups[i].faces.clear();
	}
	Triangles triFiltered;
	for (int i = 0; i < pieces.size(); ++i)
	{
		for (int j = 0; j < pieces[i].size(); ++j)
		{
			Triangle t;
			t = triangles[pieces[i][j]];
			triFiltered.push_back(t);
		}
	}
	for (int i = 0; i < triFiltered.size(); ++i)
	{
		int groupId = triFiltered[i].groupId;
		Face f;
		for (int j = 0; j < 3; ++j)
		{
			f.vertex_ids[j] = triFiltered[i].vertex_ids[j];
			f.texcoord_ids[j] = triFiltered[i].texcoord_ids[j];
			f.normal_ids[j] = triFiltered[i].normal_ids[j];
		}
		groups[groupId].faces.push_back(f);
	}
}
/*
created at 02/25/2019
modified at 02/26/2019
*/
void ObjModel::refiningBoundary(std::vector<Edge> &boundary, Triangles &triangles, std::vector<Triangle> &newTries)
{
	for (int i = 0; i < boundary.size()-1; ++i)
	{
		float angle = 2*PAI;
		for (int j = 0; j < triangles.size(); ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				if (boundary[i].vertex_ids[1] == triangles[j].vertex_ids[k])
				{
					int vtIds[3];
					for (int n = 0; n < 3; ++n)
					{
						vtIds[n] = triangles[j].vertex_ids[(k+n)%3];
					}
					math::Vec3f vec1, vec2;
					vec1 = vertices[vtIds[1]] - vertices[vtIds[0]];
					vec2 = vertices[vtIds[2]] - vertices[vtIds[0]];
					vec1.normalize();
					vec2.normalize();
					angle -= std::acosf(vec1.dot(vec2));
					break;
				}
			}
		}
		if (angle < RAD1)
		{
			Triangle t;
			t.groupId = boundary[i].groupId;
			t.vertex_ids[0] = boundary[i].vertex_ids[0];
			t.vertex_ids[1] = boundary[i].vertex_ids[1];
			t.vertex_ids[2] = boundary[i+1].vertex_ids[1];
			t.normal_ids[0] = boundary[i].normal_ids[0];
			t.normal_ids[1] = boundary[i].normal_ids[1];
			t.normal_ids[2] = boundary[i + 1].normal_ids[1];
			t.texcoord_ids[0] = boundary[i].texcoord_ids[0];
			t.texcoord_ids[1] = boundary[i].texcoord_ids[1];
			t.texcoord_ids[2] = boundary[i].texcoord_ids[2];
			newTries.push_back(t);
			triangles.push_back(t);
			Edge newEdge;
			newEdge.vertex_ids[0] = boundary[i].vertex_ids[0];
			newEdge.vertex_ids[1] = boundary[i+1].vertex_ids[1];
			newEdge.face_id = triangles.size() - 1;
			newEdge.groupId = boundary[i].groupId;
			newEdge.normal_ids[0] = boundary[i].normal_ids[0];
			newEdge.normal_ids[1] = boundary[i+1].normal_ids[1];
			newEdge.texcoord_ids[0] = boundary[i].texcoord_ids[0];
			newEdge.texcoord_ids[1] = boundary[i].texcoord_ids[1];
			newEdge.texcoord_ids[2] = boundary[i].texcoord_ids[2];
			boundary[i] = newEdge;
			boundary.erase(boundary.begin() + i+1);
			continue;
		}
		if (angle > RAD1 && angle < RAD2)
		{
			math::Vec3f vec0 = vertices[boundary[i].vertex_ids[0]] - vertices[boundary[i].vertex_ids[1]];
			math::Vec3f vec1 = vertices[boundary[i+1].vertex_ids[1]] - vertices[boundary[i+1].vertex_ids[0]];
			vec0 = (vec0 + vec1) / 2;
			math::Vec3f newVert = vertices[boundary[i].vertex_ids[1]] + vec0;
			vertices.push_back(newVert);
			math::Vec3f newNormal = (normals[boundary[i].normal_ids[0]] + normals[boundary[i].normal_ids[1]] + normals[boundary[i + 1].normal_ids[1]]) / 3;
			normals.push_back(newNormal);
			Triangle t;
			t.groupId = boundary[i].groupId;
			t.vertex_ids[0] = boundary[i].vertex_ids[0];
			t.vertex_ids[1] = boundary[i].vertex_ids[1];
			t.vertex_ids[2] = vertices.size() - 1;
			t.normal_ids[0] = boundary[i].normal_ids[0];
			t.normal_ids[1] = boundary[i].normal_ids[1];
			t.normal_ids[2] = normals.size()-1;
			t.texcoord_ids[0] = boundary[i].texcoord_ids[0];
			t.texcoord_ids[1] = boundary[i].texcoord_ids[1];
			t.texcoord_ids[2] = boundary[i].texcoord_ids[2];
			newTries.push_back(t);
			triangles.push_back(t);
			t.groupId = boundary[i+1].groupId;
			t.vertex_ids[0] = boundary[i+1].vertex_ids[0];
			t.vertex_ids[1] = vertices.size() - 1;
			t.vertex_ids[2] = boundary[i+1].vertex_ids[1];
			t.normal_ids[0] = boundary[i+1].normal_ids[0];
			t.normal_ids[1] = normals.size() - 1;
			t.normal_ids[2] = boundary[i+1].normal_ids[1];
			t.texcoord_ids[0] = boundary[i+1].texcoord_ids[0];
			t.texcoord_ids[1] = boundary[i+1].texcoord_ids[1];
			t.texcoord_ids[2] = boundary[i+1].texcoord_ids[2];
			newTries.push_back(t);
			triangles.push_back(t);
			Edge e;
			e.face_id = triangles.size() - 2;
			e.groupId = boundary[i].groupId;
			e.vertex_ids[0] = boundary[i].vertex_ids[0];
			e.vertex_ids[1] = vertices.size()-1;
			e.normal_ids[0] = boundary[i].normal_ids[0];
			e.normal_ids[1] = normals.size() - 1;
			e.texcoord_ids[0] = boundary[i].texcoord_ids[0];
			e.texcoord_ids[1] = boundary[i].texcoord_ids[1];
			e.texcoord_ids[2] = boundary[i].texcoord_ids[2];
			boundary[i] = e;
			e.face_id = triangles.size() - 1;
			e.groupId = boundary[i+1].groupId;
			e.vertex_ids[0] = vertices.size() - 1;
			e.vertex_ids[1] = boundary[i + 1].vertex_ids[1];
			e.normal_ids[0] = normals.size() - 1;
			e.normal_ids[1] = boundary[i+1].normal_ids[1];
			e.texcoord_ids[0] = boundary[i+1].texcoord_ids[0];
			e.texcoord_ids[1] = boundary[i+1].texcoord_ids[1];
			e.texcoord_ids[2] = boundary[i+1].texcoord_ids[2];
			boundary[i+1] = e;
			continue;
		}
	}
}
/*
created at 02/27/2019
when size of boundary is 3, this code can not work.
*/
void ObjModel::getHoleFillSamples(std::vector<Edge> &boundary, std::vector<math::Vec3f> &samples)
{
	float avgLen = 0.0f;
	for (int i = 0; i < boundary.size(); ++i)
	{
		math::Vec3f vec = vertices[boundary[i].vertex_ids[1]] - vertices[boundary[i].vertex_ids[0]];
		avgLen += vec.norm();
	}
	avgLen /= boundary.size();
	std::vector<math::Vec3c> vts;
	for (int i = 0; i < boundary.size(); ++i)
	{
		vts.push_back(vertices[boundary[i].vertex_ids[0]]);
	}
	float sum = 0.0f;
	float maxArea = 0.0f;
	std::vector<math::Vec3f> newVertices;
	int n = vts.size();
	for (int i = 0; i < n; ++i)
	{
		sum = 0.0f;
		std::vector<math::Vec3f> tmp;
		for (int j = 1; j < n/2-1; ++j)
		{
			int n1 = (i + j)%n;
			int n2 = (i-j+n)%n;
			math::Vec3f vec = vertices[n2] - vertices[n1];
			sum += vec.norm();
			tmp.push_back(vertices[n1]);
			tmp.push_back(vertices[n2]);
		}
		if (maxArea < sum)
		{
			maxArea = sum;
			newVertices.clear();
			for (int j = 0; j < tmp.size(); ++j)
			{
				newVertices.push_back(tmp[j]);
			}
		}
	}
	float r = avgLen / 2;
	for (int i = 0; i < newVertices.size()-1; ++i)
	{
		math::Vec3f vec = newVertices[2 * i + 1] - newVertices[2 * i];
		int n = vec.norm() / r;
		vec.normalize();
		for (int j = 0; j < n; ++j)
		{
			math::Vec3f v = newVertices[2 * i] + j*r*vec;
			samples.push_back(v);
		}
	}
	for (int i = 0; i < vts.size(); ++i)
	{
		samples.push_back(vts[i]);
	}
}
/**/