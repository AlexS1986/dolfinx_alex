
import dolfinx as dlfx
import numpy as np
import ufl
import sys
import basix.ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import gmsh
#import meshio
#import math

from pathlib import Path


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Mesh:
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind):
        self.L = L
        self.H = H
        self.W = W
        self.NL = NL
        #self.t_z = t_z
        #self.NH = NH
        #self.NW = NW
        self.n_i = n_i
        self.inclusions = inclusions
        self.n_v = n_v
        self.voids = voids
        self.Hexa = Hexa
        self.MeshName = MeshName
        self.MeshFilename = MeshFilename
        # self.XdmfMeshFilename = XdmfMeshFilename
        # self.XdmfLineFilename = XdmfLineFilename
        self.matrix_marker = matrix_marker
        self.inclusion_marker = inclusion_marker
        self.inclusion_surface_marker = inclusion_surface_marker
        self.Hertzian = Hertzian
        self.R_ind = R_ind
        #self.Void = Void


    def mesh_refinement(self, n_ref: float, scope: str):
        h_el = self.L/self.NL
        r_eff_list = []
        for i in range(0, self.n_i):
            r_eff_0 = self.inclusions[i]['stretch_factor'][0]*self.inclusions[i]['length']  #  f_i[i][0]*self.R_i[i]
            r_eff_list.append(r_eff_0)
            r_eff_1 = self.inclusions[i]['stretch_factor'][1]*self.inclusions[i]['length']
            r_eff_list.append(r_eff_1)
            r_eff_2 = self.inclusions[i]['stretch_factor'][2]*self.inclusions[i]['length']
            r_eff_list.append(r_eff_2)
        if self.n_i != 0:
            max_r = max(r_eff_list)
        
        y_pos_list = []
        for i in range(0, self.n_i):
            y_pos = self.inclusions[i]['center'][1] + self.inclusions[i]['stretch_factor'][1]*self.inclusions[i]['length']              #center_i[i][1] + self.f_i[i][1]*self.R_i[i]
            y_pos_list.append(y_pos)
        if self.n_i != 0:
            y_max = max(y_pos_list)
        else:
            y_max = 2.0*self.H/2
        
        y_neg_list = []
        for i in range(0, self.n_i):
            y_neg = self.inclusions[i]['center'][1] - self.inclusions[i]['stretch_factor'][1]*self.inclusions[i]['length'] #self.center_i[i][1] - self.f_i[i][1]*self.R_i[i]
            y_neg_list.append(y_neg)
        if self.n_i != 0:
            y_min = min(y_neg_list)
        else:
            y_min = -2.0*self.H/2

        if scope == "full":
            gmsh.model.mesh.field.add("Box", 10*(self.n_i+1))
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VIn", h_el/n_ref)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VOut", h_el)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMin", -2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMax", 2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMin", -self.H/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMax", self.H/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMin", -2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMax", 2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "Thickness", 2.0*self.W)
            gmsh.model.mesh.field.setAsBackgroundMesh(10*(self.n_i+1))

        elif scope == "dynamic":
            gmsh.model.mesh.field.add("Box", 10*(self.n_i+1))
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VIn", h_el/n_ref)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VOut", h_el)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMin", -2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMax", 2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMin", 1.1*y_min)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMax", 1.1*y_max)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMin", -2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMax", 2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "Thickness", 2.0*self.W)
            gmsh.model.mesh.field.setAsBackgroundMesh(10*(self.n_i+1))

        elif scope == "dynamic_larger":
            gmsh.model.mesh.field.add("Box", 10*(self.n_i+1))
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VIn", h_el/n_ref)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VOut", h_el)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMin", -2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMax", 2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMin", y_min-max_r)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMax", y_max+max_r)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMin", -2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMax", 2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "Thickness", 2.0*self.W)
            gmsh.model.mesh.field.setAsBackgroundMesh(10*(self.n_i+1))
        
        elif scope == "middle_section":
            gmsh.model.mesh.field.add("Box", 10*(self.n_i+1))
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VIn", h_el/n_ref)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "VOut", h_el)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMin", -2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "XMax", 2.0*self.L/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMin", -1.5*self.voids[(0,0,0)]['length'])
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "YMax", 1.5*self.voids[(0,0,0)]['length'])
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMin", -2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "ZMax", 2.0*self.W/2)
            gmsh.model.mesh.field.setNumber(10*(self.n_i+1), "Thickness", 2.0*self.W)
            gmsh.model.mesh.field.setAsBackgroundMesh(10*(self.n_i+1))
        
        gmsh.model.occ.synchronize()

    def mesh_refinement_Hertzian(self, n_ref:float):  #TODO: should be above + contact zone refinement
        h_el = self.L/self.NL
        top_center = gmsh.model.occ.addPoint(0.0, self.H/2, 0.0)  
        #top_center_line = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(0.0, self.H/2, -self.W/2), gmsh.model.occ.addPoint(0.0, self.H/2, self.W/2))
        gmsh.model.mesh.field.add("Distance", 10*(self.n_i+2))
        gmsh.model.mesh.field.setNumbers(10*(self.n_i+2), "PointsList", [top_center])
        #gmsh.model.mesh.field.setNumbers(10*(self.n_i+2), "CurvesList", [top_center_line])
        gmsh.model.mesh.field.add("Threshold", 10*(self.n_i+3))
        gmsh.model.mesh.field.setNumber(10*(self.n_i+3), "InField", 10*(self.n_i+2))
        gmsh.model.mesh.field.setNumber(10*(self.n_i+3), "SizeMin", h_el/n_ref)
        gmsh.model.mesh.field.setNumber(10*(self.n_i+3), "SizeMax", h_el)
        gmsh.model.mesh.field.setNumber(10*(self.n_i+3), "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(10*(self.n_i+3), "DistMax", 1.5*self.R_ind)
        gmsh.model.mesh.field.add("Min", 10*(self.n_i+4))
        gmsh.model.mesh.field.setNumbers(10*(self.n_i+4), "FieldsList", [10*(self.n_i+1), 10*(self.n_i+3)])
        gmsh.model.mesh.field.setAsBackgroundMesh(10*(self.n_i+4))


    


class MatrixInclusion3D(Mesh):    
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian=False, R_ind=0.0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind)
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, 0.0)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, 0.0)
            gmsh.model.occ.synchronize()

            middle_surface_line_1 = gmsh.model.occ.addLine(middle_surface_point_1, middle_surface_point_2)
            gmsh.model.occ.synchronize()
            
            matrix_volume = gmsh.model.occ.addBox(-self.L/2, -self.H/2, -self.W/2, self.L, self.H, self.W)
            print("matrix_volume = ", matrix_volume)
            gmsh.model.occ.synchronize()
                                    
            inclusion_volume_list = []
            for i in range(0, self.n_i):
                if self.inclusions[i]['shape'] == 'ellipsoid':
                    inclusion_volume = gmsh.model.occ.addSphere(self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['length'])
                    print("inclusion_volume (ellipsoid) = ", inclusion_volume)
                elif self.inclusions[i]['shape'] == 'rectangle':
                    inclusion_volume = gmsh.model.occ.addBox(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][2]-0.5*self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'])
                    print("inclusion_volume (rectangle) = ", inclusion_volume)
                gmsh.model.occ.dilate([(3, inclusion_volume)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['stretch_factor'][0], self.inclusions[i]['stretch_factor'][1], self.inclusions[i]['stretch_factor'][2])
                gmsh.model.occ.rotate([(3, inclusion_volume)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['rotation_axis'][0], self.inclusions[i]['rotation_axis'][1], self.inclusions[i]['rotation_axis'][2], self.inclusions[i]['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_volume_list.append(inclusion_volume)
            print("inclusion_volume_list = ", inclusion_volume_list)
            inclusion_volume_dimTags_list = [(3, inclusion_volume_list[i]) for i in range(self.n_i)]

            if self.n_v != 0:
                void_volume_list = []
                for v in range(0, self.n_v):
                    if self.voids[v]['shape'] == 'ellipsoid':
                            void_volume = gmsh.model.occ.addSphere(self.voids[v]['center'][0], self.voids[v]['center'][1], self.voids[v]['center'][2], self.voids[v]['length'])
                            # void_geo = gmsh.model.occ.addSurfaceLoop([void_shape])
                            # void_volume = gmsh.model.occ.addVolume([void_geo])
                            print("void_volume (ellipsoid) = ", void_volume)
                    elif self.voids[v]['shape'] == 'rectangle':
                            void_volume = gmsh.model.occ.addBox(self.voids[v]['center'][0]-0.5*self.voids[v]['length'], self.voids[v]['center'][1]-0.5*self.voids[v]['length'], self.voids[v]['center'][2]-0.5*self.voids[v]['length'], self.voids[v]['length'], self.voids[v]['length'], self.voids[v]['length'])
                            print("void_volume (rectangle) = ", void_volume)
                    gmsh.model.occ.dilate([(3, void_volume)], self.voids[v]['center'][0], self.voids[v]['center'][1], self.voids[v]['center'][2], self.voids[v]['stretch_factor'][0], self.voids[v]['stretch_factor'][1], self.voids[v]['stretch_factor'][2])
                    gmsh.model.occ.rotate([(3, void_volume)], self.voids[v]['center'][0], self.voids[v]['center'][1], self.voids[v]['center'][2], self.voids[v]['rotation_axis'][0], self.voids[v]['rotation_axis'][1], self.voids[v]['rotation_axis'][2], self.voids[v]['rotation_angle'])
                    gmsh.model.occ.synchronize()
                    void_volume_list.append(void_volume)
                void_volume_dimTags_list = [(3, void_volume_list[v]) for v in range(self.n_v)]
            
            gmsh.model.occ.synchronize()
            print('inclusion_volume_dimTags_list = ', inclusion_volume_dimTags_list)
            if self.n_v != 0:
                temp_domain_raw = gmsh.model.occ.fragment([(3, matrix_volume)], void_volume_dimTags_list, removeTool=True)
                temp_domain_tag = temp_domain_raw[1][0][0][1]
                gmsh.model.occ.synchronize()
                print("temp_domain_raw = ", temp_domain_raw)
            else:
                temp_domain_tag = matrix_volume
                gmsh.model.occ.synchronize()
            print("temp_domain_tag = ", temp_domain_tag)
            
            final_domain = gmsh.model.occ.fragment([(3, temp_domain_tag)], inclusion_volume_dimTags_list, removeTool=False)
            gmsh.model.occ.synchronize()
            print("final_domain = ", final_domain)
            gmsh.model.occ.synchronize()
            print("matrix marker: ", self.matrix_marker)
            print("inclusion marker: ", self.inclusion_marker)
            
            if self.n_i + self.n_v == 0:
                gmsh.model.addPhysicalGroup(3, [matrix_volume], self.matrix_marker) # [temp_domain_tag]
            else:
                gmsh.model.addPhysicalGroup(3, [final_domain[0][-1][1]], self.matrix_marker) # [temp_domain_tag]
                        
                for i in range(self.n_i):
                    gmsh.model.addPhysicalGroup(3, [inclusion_volume_list[i]], self.inclusion_marker[i])
            

            #gmsh.model.addPhysicalGroup(2, split_inclusion_domain_tags, self.inclusion_marker)
            gmsh.model.occ.synchronize()
            
            if self.Hertzian:
                self.mesh_refinement_Hertzian(n_ref)
            else:
                self.mesh_refinement(n_ref, "full") #n_ref
            
            if self.Hexa: #TODO
                # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # blossom, full-quad
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                # gmsh.option.setNumber("Mesh.Recombination3DLevel", 0) # 0 = Hex
                gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                    
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(self.MeshFilename)
        inclusion_volume_list.clear()
        gmsh.finalize()
        return gmsh.model


class MatrixInclusionPlaneStrain(Mesh):    
    # def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian=False, R_ind=0.0):
    #     super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind)

    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian=False, R_ind=0.0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind)
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, 0.0)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, 0.0)
            gmsh.model.occ.synchronize()

            # t = 1e-2  # small thickness for plane strain

            middle_surface_line_1 = gmsh.model.occ.addLine(middle_surface_point_1, middle_surface_point_2)
            gmsh.model.occ.synchronize()
            
            matrix_geo = gmsh.model.occ.addRectangle(-self.L/2, -self.H/2, -self.W/2, self.L, self.H)
            gmsh.model.occ.synchronize()
            print("matrix_geo = ", matrix_geo)
            
            matrix_area = gmsh.model.occ.addPlaneSurface([matrix_geo])
            gmsh.model.occ.synchronize()
            print("matrix_area = ", matrix_area)
            gmsh.model.occ.synchronize()

            inclusion_surface_list = []
            for i in range(0, self.n_i):
                if self.inclusions[i]['shape'] == 'ellipsoid':
                    inclusion_area = gmsh.model.occ.addCircle(self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], -t/2, self.inclusions[i]['length'])
                    inclusion_geo = gmsh.model.occ.addCurveLoop([inclusion_area])
                    inclusion_surface = gmsh.model.occ.addPlaneSurface([inclusion_geo])
                    gmsh.model.occ.synchronize()
                    print("inclusion_surface (ellipsoid) = ", inclusion_surface)
                elif self.inclusions[i]['shape'] == 'rectangle':
                    inclusion_surface = gmsh.model.occ.addRectangle(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], -self.W/2, self.inclusions[i]['length'], self.inclusions[i]['length'])
                    # inclusion_geo = gmsh.model.occ.addCurveLoop([inclusion_area])
                    # inclusion_surface = gmsh.model.occ.addPlaneSurface([inclusion_geo])
                    gmsh.model.occ.synchronize()
                    print("inclusion_surface (rectangle) = ", inclusion_surface)
                gmsh.model.occ.dilate([(2, inclusion_surface)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['stretch_factor'][0], self.inclusions[i]['stretch_factor'][1], self.inclusions[i]['stretch_factor'][2])
                gmsh.model.occ.rotate([(2, inclusion_surface)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['rotation_axis'][0], self.inclusions[i]['rotation_axis'][1], self.inclusions[i]['rotation_axis'][2], self.inclusions[i]['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_surface_list.append(inclusion_surface)
            print("inclusion_surface_list = ", inclusion_surface_list)
            inclusion_surface_dimTags_list = [(2, inclusion_surface_list[i]) for i in range(self.n_i)]
            print("inclusion_surface_dimTags_list = ", inclusion_surface_dimTags_list)

            if self.n_v != 0:
                void_surface_list = []
                for v in range(0, self.n_v):
                    if self.voids[v]['shape'] == 'ellipsoid':
                        void_area = gmsh.model.occ.addCircle(self.voids[v]['center'][0], self.voids[v]['center'][1], -t/2, self.voids[v]['length'])
                        void_geo = gmsh.model.occ.addCurveLoop([void_area])
                        void_surface = gmsh.model.occ.addPlaneSurface([void_geo])
                        gmsh.model.occ.synchronize()
                        print("void_surface (ellipsoid) = ", void_surface)
                    elif self.voids[v]['shape'] == 'rectangle':
                        void_surface = gmsh.model.occ.addRectangle(self.voids[v]['center'][0]-0.5*self.voids[v]['length'], self.voids[v]['center'][1]-0.5*self.voids[v]['length'], -t/2, self.voids[v]['length'], self.voids[v]['length'])
                        # void_geo = gmsh.model.occ.addCurveLoop([void_area])
                        # void_surface = gmsh.model.occ.addPlaneSurface([void_geo])
                        gmsh.model.occ.synchronize()
                        print("void_surface (rectangle) = ", void_surface)
                    gmsh.model.occ.dilate([(2, void_surface)], self.voids[v]['center'][0], self.voids[v]['center'][1], self.voids[v]['center'][2], self.voids[v]['stretch_factor'][0], self.voids[v]['stretch_factor'][1], self.voids[v]['stretch_factor'][2])
                    gmsh.model.occ.rotate([(2, void_surface)], self.voids[v]['center'][0], self.voids[v]['center'][1], self.voids[v]['center'][2], self.voids[v]['rotation_axis'][0], self.voids[v]['rotation_axis'][1], self.voids[v]['rotation_axis'][2], self.voids[v]['rotation_angle'])
                    gmsh.model.occ.synchronize()
                    void_surface_list.append(void_surface)
                void_surface_dimTags_list = [(2, void_surface_list[v]) for v in range(self.n_v)]
                print("void_surface_dimTags_list = ", void_surface_dimTags_list)
            gmsh.model.occ.synchronize()

            if self.n_v != 0:
                matrix_void_domain_2D = gmsh.model.occ.fragment([(2, matrix_area)], void_surface_dimTags_list, removeTool=True)
                gmsh.model.occ.synchronize()
                print("matrix_domain_2D = ", matrix_void_domain_2D)
                matrix_domain_3D = gmsh.model.occ.extrude([matrix_void_domain_2D[1][0][0]], 0.0, 0.0, self.W, numElements=[1])
                gmsh.model.occ.synchronize()
            else:
                matrix_domain_3D = gmsh.model.occ.extrude([(2, matrix_area)], 0.0, 0.0, self.W, numElements=[1])
                gmsh.model.occ.synchronize()
            print("matrix_domain_3D = ", matrix_domain_3D)
            matrix_domain_3D_dimTags_list = []
            for el in matrix_domain_3D:
                if el[0] == 3:
                    matrix_domain_3D_dimTags_list.append(el)
                    matrix_domain_3D_tag = el[1]
            gmsh.model.occ.synchronize()
            print("matrix_domain_3D_dimTags_list = ", matrix_domain_3D_dimTags_list)
            
            inclusion_domain_3D = []
            inclusion_domain_3D_tags_list = []
            for i in range(self.n_i):
                # gmsh.model.occ.extrude(inclusion_surface_dimTags_list, 0.0, 0.0, t, numElements=[1])
                # gmsh.model.occ.synchronize()
                inclusion_domain_3D.append(gmsh.model.occ.extrude([inclusion_surface_dimTags_list[i]], 0.0, 0.0, self.W, numElements=[1]))
                gmsh.model.occ.synchronize()
            print("inclusion_domain_3D = ", inclusion_domain_3D)
            inclusion_domain_3D_dimTags_list = []
            for el in inclusion_domain_3D:
                for sub_el in el:
                    if sub_el[0] == 3:
                        inclusion_domain_3D_dimTags_list.append(sub_el)
                        inclusion_domain_3D_tags_list.append(sub_el[1])
            gmsh.model.occ.synchronize()
            print("inclusion_domain_3D_dimTags_list = ", inclusion_domain_3D_dimTags_list)
            print("inclusion_domain_3D_tags_list = ", inclusion_domain_3D_tags_list)
            
            final_domain = gmsh.model.occ.fragment(matrix_domain_3D_dimTags_list, inclusion_domain_3D_dimTags_list, removeTool=False)
            gmsh.model.occ.synchronize()
            print("final_domain = ", final_domain)
            if self.n_i + self.n_v == 0:
                final_domain_matrix_tag = matrix_domain_3D_dimTags_list[0][1]#final_domain[0][-1][1]
            else:
                final_domain_matrix_tag = final_domain[1][0][0][1]
            gmsh.model.occ.synchronize()
            print("final_domain_matrix_tag = ", final_domain_matrix_tag)
            
            if self.n_i + self.n_v == 0:
                gmsh.model.addPhysicalGroup(3, [final_domain_matrix_tag], self.matrix_marker) 
            else:
                gmsh.model.addPhysicalGroup(3, [final_domain_matrix_tag], self.matrix_marker) 
                        
                for i in range(self.n_i):
                    gmsh.model.addPhysicalGroup(3, [inclusion_domain_3D_tags_list[i]], self.inclusion_marker[i])
            

            #gmsh.model.addPhysicalGroup(2, split_inclusion_domain_tags, self.inclusion_marker)
            gmsh.model.occ.synchronize()
            
            self.mesh_refinement(n_ref, "full") #n_ref
            
            if self.Hertzian:
                self.mesh_refinement_Hertzian(n_ref)
            
            if self.Hexa: #TODO
                #gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # blossom, full-quad
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                #gmsh.option.setNumber("Mesh.Recombination3DLevel", 0) # 0 = Hex
                #gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                    
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(self.MeshFilename)
        inclusion_surface_list.clear()
        gmsh.finalize()
        return gmsh.model




class MatrixPores3D_save(Mesh):
    def __init__(self, L, H, W, NL, n_i=0, inclusions=None, n_v=0, voids=None,
                 Hexa=False, MeshName="Mesh", MeshFilename="mesh.msh",
                 matrix_marker=1, inclusion_marker=None, inclusion_surface_marker=None,
                 Hertzian=False, R_ind=0.0,
                 n_void_x=0, n_void_y=0, n_void_z=0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa,
                         MeshName, MeshFilename, matrix_marker,
                         inclusion_marker if inclusion_marker else [],
                         inclusion_surface_marker if inclusion_surface_marker else [],
                         Hertzian, R_ind)
        self.n_void_x = n_void_x
        self.n_void_y = n_void_y
        self.n_void_z = n_void_z

    def create(self, n_ref: float = 1.0):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)

            # -------------------
            # Cube (matrix)
            # -------------------
            matrix_volume = gmsh.model.occ.addBox(-self.L/2, -self.H/2, -self.W/2,
                                                 self.L, self.H, self.W)

            # -------------------
            # Inclusions
            # -------------------
            inclusion_volumes = []
            for i in range(self.n_i):
                inc = self.inclusions[i]
                if inc["shape"] == "ellipsoid":
                    vol = gmsh.model.occ.addSphere(*inc["center"], inc["length"])
                elif inc["shape"] == "rectangle":
                    x0 = inc["center"][0] - 0.5*inc["length"]
                    y0 = inc["center"][1] - 0.5*inc["length"]
                    z0 = inc["center"][2] - 0.5*inc["length"]
                    vol = gmsh.model.occ.addBox(x0, y0, z0,
                                                inc["length"], inc["length"], inc["length"])
                gmsh.model.occ.dilate([(3, vol)], *inc["center"], *inc["stretch_factor"])
                gmsh.model.occ.rotate([(3, vol)], *inc["center"], *inc["rotation_axis"], inc["rotation_angle"])
                inclusion_volumes.append((3, vol))

            # -------------------
            # Voids
            # -------------------
            void_volumes = []
            for key, void in self.voids.items():
                if void["shape"] == "ellipsoid":
                    vol = gmsh.model.occ.addSphere(*void["center"], void["length"])
                elif void["shape"] == "rectangle":
                    x0 = void["center"][0] - 0.5*void["length"]
                    y0 = void["center"][1] - 0.5*void["length"]
                    z0 = void["center"][2] - 0.5*void["length"]
                    vol = gmsh.model.occ.addBox(x0, y0, z0,
                                                void["length"], void["length"], void["length"])
                gmsh.model.occ.dilate([(3, vol)], *void["center"], *void["stretch_factor"])
                gmsh.model.occ.rotate([(3, vol)], *void["center"], *void["rotation_axis"], void["rotation_angle"])
                void_volumes.append((3, vol))

            gmsh.model.occ.synchronize()

            # -------------------
            # Subtract inclusions and voids from matrix
            # -------------------
            cut_objects = inclusion_volumes + void_volumes
            if cut_objects:
                result = gmsh.model.occ.cut([(3, matrix_volume)], cut_objects)
                gmsh.model.occ.synchronize()
                final_matrix_tag = result[0][0][1] if result else matrix_volume
            else:
                final_matrix_tag = matrix_volume

            # -------------------
            # Physical groups
            # -------------------
            gmsh.model.addPhysicalGroup(3, [final_matrix_tag], self.matrix_marker)
            for idx, inc in enumerate(inclusion_volumes):
                if self.inclusion_marker:
                    gmsh.model.addPhysicalGroup(3, [inc[1]], self.inclusion_marker[idx])
            for void in void_volumes:
                gmsh.model.addPhysicalGroup(3, [void[1]], 99)  # void marker

            # -------------------
            # Mesh refinement using NL
            # -------------------
            self.mesh_refinement(n_ref=self.NL, scope="full")  # Use NL as mesh density factor

            # -------------------
            # Mesh generation
            # -------------------
            if self.Hexa:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(self.MeshFilename)
        gmsh.finalize()
        return gmsh.model


import gmsh
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank() if 'MPI' in globals() else 0

class MatrixPores3DAdaptive(Mesh):
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa,
                 MeshName="Mesh", MeshFilename="mesh.msh",
                 matrix_marker=1, inclusion_marker=None, inclusion_surface_marker=None,
                 Hertzian=False, R_ind=0.0, n_void_x=0, n_void_y=0, n_void_z=0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa,
                         MeshName, MeshFilename, matrix_marker, inclusion_marker,
                         inclusion_surface_marker, Hertzian, R_ind)
        self.n_void_x = n_void_x
        self.n_void_y = n_void_y
        self.n_void_z = n_void_z

    def mesh_refinement_adaptive(self, gmsh_void_tags, max_ratio: float = 20.0):
        """
        Builds Distance fields to the *surfaces* bounding each void volume,
        then a Threshold field mapping distance -> element size from h_min to h_max.
        """
        if not gmsh_void_tags:
            return

        h_min = self.L / self.NL
        h_max = h_min * max_ratio

        distance_field_ids = []
        base_id = 1000

        for i, vol_tag in enumerate(gmsh_void_tags):
            boundary = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=True)
            surface_tags = [dtag[1] for dtag in boundary if dtag[0] == 2]
            if not surface_tags:
                continue

            fid = base_id + i
            gmsh.model.mesh.field.add("Distance", fid)
            gmsh.model.mesh.field.setNumbers(fid, "SurfacesList", surface_tags)
            distance_field_ids.append(fid)

        if not distance_field_ids:
            return

        if len(distance_field_ids) == 1:
            in_field = distance_field_ids[0]
        else:
            min_fid = 2000
            gmsh.model.mesh.field.add("Min", min_fid)
            gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", distance_field_ids)
            in_field = min_fid

        thresh_fid = 3000
        gmsh.model.mesh.field.add("Threshold", thresh_fid)
        gmsh.model.mesh.field.setNumber(thresh_fid, "InField", in_field)
        gmsh.model.mesh.field.setNumber(thresh_fid, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(thresh_fid, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(thresh_fid, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh_fid, "DistMax", max(self.L, self.H, self.W) * 0.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_fid)

    def create(self, n_ref: float = 1.0):
        """
        Build geometry (matrix, inclusions, voids), subtract voids from inclusions,
        subtract voids from matrix, then fragment matrix by inclusions.
        """
        gmsh.initialize()
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)

            # --- OCC options compatible with most versions
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
            # removed Geometry.OCCConnect (unsupported)

            # --- Matrix (box)
            matrix_volume = gmsh.model.occ.addBox(-self.L/2, -self.H/2, -self.W/2,
                                                 self.L, self.H, self.W)
            gmsh.model.occ.synchronize()

            # --- Inclusions
            inclusion_volume_list = []
            for i in range(self.n_i):
                inc = self.inclusions[i]
                if inc['shape'] == 'ellipsoid':
                    inc_vol = gmsh.model.occ.addSphere(*inc['center'], inc['length'])
                elif inc['shape'] == 'rectangle':
                    c = inc['center']
                    l = inc['length']
                    inc_vol = gmsh.model.occ.addBox(c[0]-0.5*l, c[1]-0.5*l, c[2]-0.5*l, l, l, l)
                else:
                    raise ValueError(f"Unknown inclusion shape: {inc['shape']}")

                gmsh.model.occ.dilate([(3, inc_vol)], *inc['center'], *inc['stretch_factor'])
                gmsh.model.occ.rotate([(3, inc_vol)], *inc['center'], *inc['rotation_axis'], inc['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_volume_list.append(inc_vol)

            inclusion_volume_dimTags_list = [(3, v) for v in inclusion_volume_list]

            # --- Voids
            void_volume_list = []
            if self.n_void_x > 0 and self.n_void_y > 0 and self.n_void_z > 0:
                for vx in range(self.n_void_x):
                    for vy in range(self.n_void_y):
                        for vz in range(self.n_void_z):
                            try:
                                v = self.voids[vx, vy, vz]
                            except Exception:
                                v = self.voids[(vx, vy, vz)]

                            if v['shape'] == 'ellipsoid':
                                void_vol = gmsh.model.occ.addSphere(*v['center'], v['length'])
                            elif v['shape'] == 'rectangle':
                                c = v['center']
                                l = v['length']
                                void_vol = gmsh.model.occ.addBox(c[0]-0.5*l, c[1]-0.5*l, c[2]-0.5*l, l, l, l)
                            else:
                                raise ValueError(f"Unknown void shape: {v['shape']}")

                            gmsh.model.occ.dilate([(3, void_vol)], *v['center'], *v['stretch_factor'])
                            gmsh.model.occ.rotate([(3, void_vol)], *v['center'], *v['rotation_axis'], v['rotation_angle'])
                            gmsh.model.occ.synchronize()
                            void_volume_list.append(void_vol)
                            try:
                                self.voids[vx, vy, vz]['gmsh_tag'] = void_vol
                            except Exception:
                                pass
            elif hasattr(self.voids, "items"):
                for key, v in self.voids.items():
                    if v['shape'] == 'ellipsoid':
                        void_vol = gmsh.model.occ.addSphere(*v['center'], v['length'])
                    elif v['shape'] == 'rectangle':
                        c = v['center']
                        l = v['length']
                        void_vol = gmsh.model.occ.addBox(c[0]-0.5*l, c[1]-0.5*l, c[2]-0.5*l, l, l, l)
                    else:
                        raise ValueError(f"Unknown void shape: {v['shape']}")
                    gmsh.model.occ.dilate([(3, void_vol)], *v['center'], *v['stretch_factor'])
                    gmsh.model.occ.rotate([(3, void_vol)], *v['center'], *v['rotation_axis'], v['rotation_angle'])
                    gmsh.model.occ.synchronize()
                    void_volume_list.append(void_vol)
                    try:
                        self.voids[key]['gmsh_tag'] = void_vol
                    except Exception:
                        pass

            gmsh.model.occ.synchronize()

            # --- Boolean operations
            inclusion_minus_voids_dimTags_list = []
            if inclusion_volume_dimTags_list and void_volume_list:
                try:
                    inclusion_minus_voids = gmsh.model.occ.fragment(
                        inclusion_volume_dimTags_list, [(3, v) for v in void_volume_list], removeTool=True
                    )
                    gmsh.model.occ.synchronize()
                    inclusion_minus_voids_dimTags_list = [(e[0], e[1]) for e in inclusion_minus_voids[0] if e[0] == 3]
                except Exception:
                    inclusion_minus_voids_dimTags_list = inclusion_volume_dimTags_list

            matrix_minus_volume_tag = matrix_volume
            if void_volume_list:
                try:
                    matrix_minus_voids = gmsh.model.occ.fragment([(3, matrix_volume)], [(3, v) for v in void_volume_list], removeTool=True)
                    gmsh.model.occ.synchronize()
                    matrix_minus_volume_tag = [e for e in matrix_minus_voids[0] if e[0] == 3][0][1]
                except Exception:
                    matrix_minus_volume_tag = matrix_volume

            final_domain_matrix_tag = matrix_minus_volume_tag
            if inclusion_minus_voids_dimTags_list:
                try:
                    final_fragment = gmsh.model.occ.fragment([(3, matrix_minus_volume_tag)], inclusion_minus_voids_dimTags_list, removeTool=False)
                    gmsh.model.occ.synchronize()
                    final_domain_matrix_tag = final_fragment[-1][0][0][1]
                except Exception:
                    pass

            # --- Physical groups
            gmsh.model.addPhysicalGroup(3, [final_domain_matrix_tag], self.matrix_marker)
            inc_tags_to_add = [t[1] for t in inclusion_minus_voids_dimTags_list] if inclusion_minus_voids_dimTags_list else inclusion_volume_list
            for i, inc_tag in enumerate(inc_tags_to_add):
                if self.inclusion_marker and i < len(self.inclusion_marker):
                    try:
                        gmsh.model.addPhysicalGroup(3, [inc_tag], self.inclusion_marker[i])
                    except Exception:
                        pass
            if void_volume_list:
                gmsh.model.addPhysicalGroup(3, void_volume_list, 99)

            # --- Adaptive refinement
            self.mesh_refinement_adaptive(gmsh_void_tags=void_volume_list, max_ratio=20.0)

            # --- Hexahedral options
            if self.Hexa:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

        # --- Mesh generation
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(self.MeshFilename)
        gmsh.finalize()
        return gmsh.model




    
class MatrixPores3D(Mesh):    
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian=False, R_ind=0.0, n_void_x=0, n_void_y=0, n_void_z=0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind)
        self.n_void_x = n_void_x
        self.n_void_y = n_void_y
        self.n_void_z = n_void_z
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, 0.0)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, 0.0)
            gmsh.model.occ.synchronize()

            matrix_volume = gmsh.model.occ.addBox(-self.L/2, -self.H/2, -self.W/2, self.L, self.H, self.W)
            print("matrix_volume = ", matrix_volume)
            gmsh.model.occ.synchronize()
                                    
            inclusion_volume_list = []
            corner_x = []
            corner_y = []
            corner_z = []
            for i in range(0, self.n_i):
                corner_x.append(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'])
                corner_y.append(self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'])
                corner_z.append(self.inclusions[i]['center'][2]-0.5*self.inclusions[i]['length'])
                if self.inclusions[i]['shape'] == 'ellipsoid':
                    inclusion_volume = gmsh.model.occ.addSphere(self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['length'])
                    print("inclusion_volume (ellipsoid) = ", inclusion_volume)
                elif self.inclusions[i]['shape'] == 'rectangle':
                    #inclusion_volume = gmsh.model.occ.addBox(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][2]-0.5*self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'])
                    inclusion_volume = gmsh.model.occ.addBox(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][2]-0.5*self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'], self.inclusions[i]['length'])
                    print("inclusion_volume (rectangle) = ", inclusion_volume)
                gmsh.model.occ.dilate([(3, inclusion_volume)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['stretch_factor'][0], self.inclusions[i]['stretch_factor'][1], self.inclusions[i]['stretch_factor'][2])
                gmsh.model.occ.rotate([(3, inclusion_volume)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], self.inclusions[i]['center'][2], self.inclusions[i]['rotation_axis'][0], self.inclusions[i]['rotation_axis'][1], self.inclusions[i]['rotation_axis'][2], self.inclusions[i]['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_volume_list.append(inclusion_volume)
            print("inclusion_volume_list = ", inclusion_volume_list)
            inclusion_volume_dimTags_list = [(3, inclusion_volume_list[i]) for i in range(self.n_i)]
            print('inclusion_volume_dimTags_list = ', inclusion_volume_dimTags_list)

            void_volume_list = []
            for v_x in range(0, self.n_void_x):
                for v_y in range(0, self.n_void_y):
                    for v_z in range(0, self.n_void_z):
                        if self.voids[v_x, v_y, v_z]['shape'] == 'ellipsoid':
                            void_volume = gmsh.model.occ.addSphere(self.voids[v_x, v_y, v_z]['center'][0], self.voids[v_x, v_y, v_z]['center'][1], self.voids[v_x, v_y, v_z]['center'][2], self.voids[v_x, v_y, v_z]['length'])
                            # void_geo = gmsh.model.occ.addSurfaceLoop([void_shape])
                            # void_volume = gmsh.model.occ.addVolume([void_geo])
                            print("void_volume (ellipsoid) = ", void_volume)
                        elif self.voids[v_x, v_y, v_z]['shape'] == 'rectangle':
                            void_volume = gmsh.model.occ.addBox(self.voids[v_x, v_y, v_z]['center'][0]-0.5*self.voids[v_x, v_y, v_z]['length'], self.voids[v_x, v_y, v_z]['center'][1]-0.5*self.voids[v_x, v_y, v_z]['length'], self.voids[v_x, v_y, v_z]['center'][2]-0.5*self.voids[v_x, v_y, v_z]['length'], self.voids[v_x, v_y, v_z]['length'], self.voids[v_x, v_y, v_z]['length'], self.voids[v_x, v_y, v_z]['length'])
                            print("void_volume (rectangle) = ", void_volume)
                        gmsh.model.occ.dilate([(3, void_volume)], self.voids[v_x, v_y, v_z]['center'][0], self.voids[v_x, v_y, v_z]['center'][1], self.voids[v_x, v_y, v_z]['center'][2], self.voids[v_x, v_y, v_z]['stretch_factor'][0], self.voids[v_x, v_y, v_z]['stretch_factor'][1], self.voids[v_x, v_y, v_z]['stretch_factor'][2])
                        gmsh.model.occ.rotate([(3, void_volume)], self.voids[v_x, v_y, v_z]['center'][0], self.voids[v_x, v_y, v_z]['center'][1], self.voids[v_x, v_y, v_z]['center'][2], self.voids[v_x, v_y, v_z]['rotation_axis'][0], self.voids[v_x, v_y, v_z]['rotation_axis'][1], self.voids[v_x, v_y, v_z]['rotation_axis'][2], self.voids[v_x, v_y, v_z]['rotation_angle'])
                        gmsh.model.occ.synchronize()
                        void_volume_list.append(void_volume)
            void_volume_dimTags_list = [(3, void_volume_list[v]) for v in range(self.n_v)]
            print("void_volume_dimTags_list = ", void_volume_dimTags_list)
                        
            gmsh.model.occ.synchronize()
                        
            #matrix_voids = gmsh.model.occ.fragment([(3, matrix_volume)], void_volume_dimTags_list, removeTool=True)
            matrix_minus_inclusion = gmsh.model.occ.fragment([(3, matrix_volume)], inclusion_volume_dimTags_list, removeTool=False)
            gmsh.model.occ.synchronize()
            print("matrix_minus_inclusion = ", matrix_minus_inclusion)
            #print("matrix_voids = ", matrix_voids)
            #matrix_voids_tag = matrix_voids[1][0][0][1]
            gmsh.model.occ.synchronize()
            
            #print("matrix_voids_tag = ", matrix_voids_tag)
            
            # temp_domain_raw = gmsh.model.occ.fragment([(3, matrix_volume)], void_volume_dimTags_list, removeTool=True)
            matrix_minus_inclusion_tag = matrix_minus_inclusion[1][0][0][1]
            print("matrix_minus_inclusion_tag = ", matrix_minus_inclusion_tag)
            gmsh.model.occ.synchronize()
                        
            inclusion_minus_voids = gmsh.model.occ.fragment(inclusion_volume_dimTags_list, void_volume_dimTags_list, removeTool=True)
            gmsh.model.occ.synchronize()
            print("inclusion_minus_voids = ", inclusion_minus_voids)
            inclusion_minus_voids_tags = inclusion_minus_voids[-1][0][0][1]
            inclusion_minus_voids_dimTags_list = [(3, inclusion_minus_voids_tags)]
            # for el in inclusion_minus_voids[0]:
            #     if el[1] > matrix_minus_inclusion_tag:
            #         inclusion_minus_voids_dimTags_list.append(el)
            #         inclusion_minus_voids_tags = el[1]
            print("inclusion_minus_voids_tags = ", inclusion_minus_voids_tags)
            print("inclusion_minus_voids_dimTags_list = ", inclusion_minus_voids_dimTags_list)
            
            final_domain = gmsh.model.occ.fragment([(3, matrix_minus_inclusion_tag)], inclusion_minus_voids_dimTags_list, removeTool=False)
            gmsh.model.occ.synchronize()
            gmsh.model.occ.synchronize()
            print("final_domain = ", final_domain)
            print("matrix marker: ", self.matrix_marker)
            print("inclusion marker: ", self.inclusion_marker)
                        
            gmsh.model.addPhysicalGroup(3, [final_domain[-1][0][0][1]], self.matrix_marker) # [temp_domain_tag]
            # gmsh.model.addPhysicalGroup(3, [matrix_voids_tag], self.matrix_marker) # [temp_domain_tag]
                        
            gmsh.model.addPhysicalGroup(3, [inclusion_minus_voids_tags], self.inclusion_marker[0])
            gmsh.model.occ.synchronize()
            
            if self.Hertzian:
                self.mesh_refinement_Hertzian(n_ref)
            else:
                self.mesh_refinement(n_ref, "middle_section") #n_ref
            
            # if self.Hexa: #TODO
            #     # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # blossom, full-quad
            #     gmsh.option.setNumber("Mesh.RecombineAll", 1)
            #     # gmsh.option.setNumber("Mesh.Recombination3DLevel", 0) # 0 = Hex
            #     gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
            #     gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                    
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(self.MeshFilename)
        inclusion_volume_list.clear()
        gmsh.finalize()
        return gmsh.model


class MatrixInclusion2D(Mesh):    
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian=False, R_ind=0.0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, inclusion_surface_marker, Hertzian, R_ind)
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, 0.0)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, 0.0)
            gmsh.model.occ.synchronize()

            middle_surface_line_1 = gmsh.model.occ.addLine(middle_surface_point_1, middle_surface_point_2)
            gmsh.model.occ.synchronize()
            
            matrix_area = gmsh.model.occ.addRectangle(-self.L/2, -self.H/2, 0.0, self.L, self.H)
            print("matrix_area = ", matrix_area)
            gmsh.model.occ.synchronize()
                                    
            inclusion_area_list = []
            for i in range(0, self.n_i):
                if self.inclusions[i]['shape'] == 'ellipsoid':
                    inclusion_shape = gmsh.model.occ.addCircle(self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['length'])
                    inclusion_geo = gmsh.model.occ.addCurveLoop([inclusion_shape])
                    inclusion_area = gmsh.model.occ.addPlaneSurface([inclusion_geo])
                    print("inclusion_area (ellipsoid) = ", inclusion_area)
                elif self.inclusions[i]['shape'] == 'rectangle':
                    inclusion_area = gmsh.model.occ.addRectangle(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], 0.0, self.inclusions[i]['length'], self.inclusions[i]['length'])
                    print("inclusion_area (rectangle) = ", inclusion_area)
                gmsh.model.occ.dilate([(2, inclusion_area)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['stretch_factor'][0], self.inclusions[i]['stretch_factor'][1], 1.0)
                gmsh.model.occ.rotate([(2, inclusion_area)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['rotation_axis'][0], self.inclusions[i]['rotation_axis'][1], self.inclusions[i]['rotation_axis'][2], self.inclusions[i]['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_area_list.append(inclusion_area)
            inclusion_area_dimTags_list = [(2, inclusion_area_list[i]) for i in range(self.n_i)]

            if self.n_v != 0:
                void_area_list = []
                for v in range(0, self.n_v):
                    if self.voids[v]['shape'] == 'ellipsoid':
                            void_shape = gmsh.model.occ.addCircle(self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['length'])
                            void_geo = gmsh.model.occ.addCurveLoop([void_shape])
                            void_area = gmsh.model.occ.addPlaneSurface([void_geo])
                            print("void_area (ellipsoid) = ", void_area)
                    elif self.voids[i]['shape'] == 'rectangle':
                            void_area = gmsh.model.occ.addRectangle(self.voids[v]['center'][0]-0.5*self.voids[v]['length'], self.voids[v]['center'][1]-0.5*self.voids[v]['length'], 0.0, self.voids[v]['length'], self.voids[v]['length'])
                            print("void_area (rectangle) = ", void_area)
                    gmsh.model.occ.dilate([(2, void_area)], self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['stretch_factor'][0], self.voids[v]['stretch_factor'][1], 1.0)
                    gmsh.model.occ.rotate([(2, void_area)], self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['rotation_axis'][0], self.voids[v]['rotation_axis'][1], self.voids[v]['rotation_axis'][2], self.voids[v]['rotation_angle'])
                    gmsh.model.occ.synchronize()
                    void_area_list.append(void_area)
                void_area_dimTags_list = [(2, void_area_list[v]) for v in range(self.n_v)]
            
            gmsh.model.occ.synchronize()
            print('inclusion_area_dimTags_list = ', inclusion_area_dimTags_list)
            if self.n_v != 0:
                temp_domain_raw = gmsh.model.occ.fragment([(2, matrix_area)], void_area_dimTags_list, removeTool=True)
                temp_domain_tag = temp_domain_raw[1][0][0][1]
                gmsh.model.occ.synchronize()
                print("temp_domain_raw = ", temp_domain_raw)
            else:
                temp_domain_tag = matrix_area
                gmsh.model.occ.synchronize()
            print("temp_domain_tag = ", temp_domain_tag)
            
            final_domain = gmsh.model.occ.fragment([(2, temp_domain_tag)], inclusion_area_dimTags_list, removeTool=False)
            gmsh.model.occ.synchronize()
            print("final_domain = ", final_domain)
            gmsh.model.occ.synchronize()

            if self.n_v + self.n_i == 0:
                gmsh.model.addPhysicalGroup(2, [matrix_area], self.matrix_marker) # [temp_domain_tag]
            else:           
                gmsh.model.addPhysicalGroup(2, [final_domain[0][0][1]], self.matrix_marker) # [temp_domain_tag]
            
            for i in range(self.n_i):
                gmsh.model.addPhysicalGroup(2, [inclusion_area_list[i]], self.inclusion_marker[i])


            #gmsh.model.addPhysicalGroup(2, split_inclusion_domain_tags, self.inclusion_marker)
            gmsh.model.occ.synchronize()
            
            #self.mesh_refinement(n_ref, "full")
            self.mesh_refinement(n_ref, "dynamic")
            
            if self.Hertzian:
                self.mesh_refinement_Hertzian(n_ref)
            
            if self.Hexa: #TODO
                #gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # blossom, full-quad
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                #gmsh.option.setNumber("Mesh.Recombination3DLevel", 0) # 0 = Hex
                #gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                    
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(self.MeshFilename)
        inclusion_area_list.clear()
        gmsh.finalize()
        return gmsh.model


class MatrixInclusion2D_center_line(Mesh):    
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker=1, inclusion_marker=2, Hertzian=False, R_ind=0.0):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, Hertzian, R_ind)
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, 0.0)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, 0.0)
            gmsh.model.occ.synchronize()

            middle_surface_line_1 = gmsh.model.occ.addLine(middle_surface_point_1, middle_surface_point_2)
            gmsh.model.occ.synchronize()
            
            matrix_area = gmsh.model.occ.addRectangle(-self.L/2, -self.H/2, 0.0, self.L, self.H)
            print("matrix_area = ", matrix_area)
            gmsh.model.occ.synchronize()
                                    
            inclusion_area_list = []
            for i in range(0, self.n_i):
                if self.inclusions[i]['shape'] == 'ellipsoid':
                    inclusion_shape = gmsh.model.occ.addCircle(self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['length'])
                    inclusion_geo = gmsh.model.occ.addCurveLoop([inclusion_shape])
                    inclusion_area = gmsh.model.occ.addPlaneSurface([inclusion_geo])
                    print("inclusion_area (ellipsoid) = ", inclusion_area)
                elif self.inclusions[i]['shape'] == 'rectangle':
                    inclusion_area = gmsh.model.occ.addRectangle(self.inclusions[i]['center'][0]-0.5*self.inclusions[i]['length'], self.inclusions[i]['center'][1]-0.5*self.inclusions[i]['length'], 0.0, self.inclusions[i]['length'], self.inclusions[i]['length'])
                    print("inclusion_area (rectangle) = ", inclusion_area)
                gmsh.model.occ.dilate([(2, inclusion_area)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['stretch_factor'][0], self.inclusions[i]['stretch_factor'][1], 1.0)
                gmsh.model.occ.rotate([(2, inclusion_area)], self.inclusions[i]['center'][0], self.inclusions[i]['center'][1], 0.0, self.inclusions[i]['rotation_axis'][0], self.inclusions[i]['rotation_axis'][1], self.inclusions[i]['rotation_axis'][2], self.inclusions[i]['rotation_angle'])
                gmsh.model.occ.synchronize()
                inclusion_area_list.append(inclusion_area)
            inclusion_area_dimTags_list = [(2, inclusion_area_list[i]) for i in range(self.n_i)]

            if self.n_v != 0:
                void_area_list = []
                for v in range(0, self.n_v):
                    if self.voids[v]['shape'] == 'ellipsoid':
                            void_shape = gmsh.model.occ.addCircle(self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['length'])
                            void_geo = gmsh.model.occ.addCurveLoop([void_shape])
                            void_area = gmsh.model.occ.addPlaneSurface([void_geo])
                            print("void_area (ellipsoid) = ", void_area)
                    elif self.voids[i]['shape'] == 'rectangle':
                            void_area = gmsh.model.occ.addRectangle(self.voids[v]['center'][0]-0.5*self.voids[v]['length'], self.voids[v]['center'][1]-0.5*self.voids[v]['length'], 0.0, self.voids[v]['length'], self.voids[v]['length'])
                            print("void_area (rectangle) = ", void_area)
                    gmsh.model.occ.dilate([(2, void_area)], self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['stretch_factor'][0], self.voids[v]['stretch_factor'][1], 1.0)
                    gmsh.model.occ.rotate([(2, void_area)], self.voids[v]['center'][0], self.voids[v]['center'][1], 0.0, self.voids[v]['rotation_axis'][0], self.voids[v]['rotation_axis'][1], self.voids[v]['rotation_axis'][2], self.voids[v]['rotation_angle'])
                    gmsh.model.occ.synchronize()
                    void_area_list.append(void_area)
                void_area_dimTags_list = [(2, void_area_list[v]) for v in range(self.n_v)]

            gmsh.model.occ.synchronize()
            print('inclusion_area_dimTags_list = ', inclusion_area_dimTags_list)
            if self.n_v != 0:
                void_domain_raw = gmsh.model.occ.fragment([(2, matrix_area)], void_area_dimTags_list, removeTool=True)
                void_domain_tag = void_domain_raw[1][0][0][1]
                void_domain_dimTags_list = [(2, void_domain_tag)]
                gmsh.model.occ.synchronize()
                print("void_domain_raw = ", void_domain_raw)

            else:
                void_domain_tag = matrix_area
                void_domain_dimTags_list = [(2, matrix_area)]
                gmsh.model.occ.synchronize()
            print("void_domain_tag = ", void_domain_tag)
            print("void_domain_dimTags_list = ", void_domain_dimTags_list)


            combined_matrix_area_w_center_line_raw = gmsh.model.occ.fragment(void_domain_dimTags_list, [(1, middle_surface_line_1)], removeTool=False, removeObject=True)
            combined_matrix_area_w_center_line_dimTags_list = combined_matrix_area_w_center_line_raw[1][0]
            combined_matrix_area_w_center_line_tags_list = [el[1] for el in combined_matrix_area_w_center_line_dimTags_list]
            
            gmsh.model.occ.synchronize()
            print("combined_matrix_area_w_center_line_raw = ", combined_matrix_area_w_center_line_raw)
            print("combined_matrix_area_w_center_line_dimTags_list = ", combined_matrix_area_w_center_line_dimTags_list)
            print("combined_matrix_area_w_center_line_tags_list = ", combined_matrix_area_w_center_line_tags_list)
            
            combined_inclusion_area_w_center_line_dimTags_list_per_inclusion = []
            combined_inclusion_area_w_center_line_tags_list_per_inclusion = []
            combined_inclusion_area_w_center_line_dimTags_list_total = []
            for i in range(self.n_i):
                combined_inclusion_area_w_center_line_raw = gmsh.model.occ.fragment([inclusion_area_dimTags_list[i]], [(1, middle_surface_line_1)], removeTool=False, removeObject=True)
                combined_inclusion_area_w_center_line_dimTags_list = combined_inclusion_area_w_center_line_raw[1][0]
                print("combined_inclusion_area_w_center_line_raw = ", combined_inclusion_area_w_center_line_raw)
                print("combined_inclusion_area_w_center_line_dimTags_list = ", combined_inclusion_area_w_center_line_dimTags_list)
                combined_inclusion_area_w_center_line_dimTags_list_per_inclusion.append(combined_inclusion_area_w_center_line_dimTags_list)
                combined_inclusion_area_w_center_line_tags_list = []
                for el in combined_inclusion_area_w_center_line_dimTags_list:
                    combined_inclusion_area_w_center_line_tags_list.append(el[1])
                combined_inclusion_area_w_center_line_tags_list_per_inclusion.append(combined_inclusion_area_w_center_line_tags_list)
                combined_inclusion_area_w_center_line_dimTags_list_total.extend(combined_inclusion_area_w_center_line_dimTags_list)
                gmsh.model.occ.synchronize()
            print("combined_inclusion_area_w_center_line_dimTags_list_per_inclusion = ", combined_inclusion_area_w_center_line_dimTags_list_per_inclusion)
            print("combined_inclusion_area_w_center_line_tags_list_per_inclusion = ", combined_inclusion_area_w_center_line_tags_list_per_inclusion)
            print("combined_inclusion_area_w_center_line_dimTags_list_total = ", combined_inclusion_area_w_center_line_dimTags_list_total)


            final_domain_raw = gmsh.model.occ.fragment(combined_matrix_area_w_center_line_dimTags_list, combined_inclusion_area_w_center_line_dimTags_list_total, removeTool=False)
            final_domain_dimTags_list = final_domain_raw[0]
            final_domain_tags_list = [el[1] for el in final_domain_dimTags_list]
            gmsh.model.occ.synchronize()
            print("final_domain_raw = ", final_domain_raw)
            print("final_domain_dimTags_list = ", final_domain_dimTags_list)
            print("final_domain_tags_list = ", final_domain_tags_list)
            gmsh.model.occ.synchronize()
                                    
            gmsh.model.addPhysicalGroup(2, final_domain_tags_list, self.matrix_marker)
            for i in range(self.n_i):
                gmsh.model.addPhysicalGroup(2, combined_inclusion_area_w_center_line_tags_list_per_inclusion[i], self.inclusion_marker[i])


            # gmsh.model.addPhysicalGroup(2, split_inclusion_domain_tags, self.inclusion_marker)
            gmsh.model.occ.synchronize()
            
            self.mesh_refinement(n_ref, "dynamic")
            
            if self.Hertzian:
                self.mesh_refinement_Hertzian(n_ref)
            
            if self.Hexa: #TODO
                #gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # blossom, full-quad
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                #gmsh.option.setNumber("Mesh.Recombination3DLevel", 0) # 0 = Hex
                #gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                    
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(self.MeshFilename)
        inclusion_area_list.clear()
        gmsh.finalize()
        return gmsh.model


    
class Box2D(Mesh):    
    # def __init__(self, L, H, W, NL, n_i, R_i, center_i, f_i, inclusion_rotation_axis, inclusion_rotation_angle, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker=1, inclusion_marker=2, Hertzian=False, R_ind=0.0, Void=False):
    #     super().__init__(L, H, W, NL, n_i, R_i, center_i, f_i, inclusion_rotation_axis, inclusion_rotation_angle, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker, Hertzian, R_ind, Void)
    def __init__(self, L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker,  inclusion_surface_marker, Hertzian, R_ind):
        super().__init__(L, H, W, NL, n_i, inclusions, n_v, voids, Hexa, MeshName, MeshFilename, XdmfMeshFilename, XdmfLineFilename, matrix_marker, inclusion_marker,  inclusion_surface_marker, Hertzian, R_ind)
    
    def create(self, n_ref:float):
        gmsh.initialize()
        
        if rank == 0:
            gmsh.model.add(self.MeshName)
            gmsh.model.setCurrent(self.MeshName)
            middle_surface_point_1 = gmsh.model.occ.addPoint(-self.L/2, 0.0, -self.W/2)
            middle_surface_point_2 = gmsh.model.occ.addPoint(self.L/2, 0.0, -self.W/2)
            gmsh.model.occ.synchronize()

            middle_surface_line_1 = gmsh.model.occ.addLine(middle_surface_point_1, middle_surface_point_2)
            gmsh.model.occ.synchronize()
            matrix_area = gmsh.model.occ.addRectangle(-self.L/2, -self.H/2, 0.0, self.L, self.H)
            print("matrix_area = ", matrix_area)
            print("matrix_area shape= ", np.shape(matrix_area))
            gmsh.model.occ.synchronize()
            matrix_w_center_line = gmsh.model.occ.fragment([(2, matrix_area)], [(1, middle_surface_line_1)], removeTool=True)
            gmsh.model.occ.synchronize()
            print("matrix_w_center_line = ", matrix_w_center_line)
                                    
            gmsh.model.addPhysicalGroup(2, [matrix_area], self.matrix_marker)
            gmsh.model.occ.synchronize()

            self.mesh_refinement(n_ref, "full")

            gmsh.model.mesh.generate(2)
            gmsh.write(self.MeshFilename)
            gmsh.finalize()
        return gmsh.model
    

