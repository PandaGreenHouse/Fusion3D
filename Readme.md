	//dmrecon
	makescene -i <image-dir> <scene-dir>
	sfmrecon <scene-dir>
	dmrecon -s2 <scene-dir>
	scene2pset -F2 <scene-dir> <scene-dir>/pset-L2.ply
	fssrecon <scene-dir>/pset-L2.ply <scene-dir>/surface-L2.ply
	meshclean -t10 <scene-dir>/surface-L2.ply <scene-dir>/surface-L2-clean.ply

	
	//smvs
	makescene -i <image-dir> <scene-dir>
	sfmrecon <scene-dir>
	smvsrecon <scene-dir>
	fssrecon <scene-dir>/smvs-[B,S].ply <scene-dir>/smvs-surface.ply
	meshclean -p10 <scene-dir>/smvs-surface.ply <scene-dir>/smvs-clean.ply
"This is 3D reconstruction pipeline" 
