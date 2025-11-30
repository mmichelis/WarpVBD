### PBRT Implementation from DiffPD (https://diffpd.csail.mit.edu)

from pathlib import Path
import shutil
import os

import numpy as np


def create_folder(folder_name, exist_ok=False):
    if not exist_ok and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=exist_ok)


# This class assumes z is pointing up.
class PbrtRenderer(object):
    def __init__(self, options=None):
        self.__temporary_folder = Path('.tmp')
        create_folder(self.__temporary_folder)
        if options is None: options = {}

        # Image metadata.
        file_name = options['file_name'] if 'file_name' in options else 'output.exr'
        file_name = str(file_name)
        assert file_name.endswith('.png') or file_name.endswith('.exr')
        file_name_only = file_name[:-4]
        self.__file_name_only = file_name_only

        resolution = options['resolution'] if 'resolution' in options else (800, 800)
        resolution = tuple(resolution)
        assert len(resolution) == 2
        resolution = [int(r) for r in resolution]
        self.__resolution = tuple(resolution)

        sample = options['sample'] if 'sample' in options else 4
        sample = int(sample)
        assert sample > 0
        self.__sample = sample

        max_depth = options['max_depth'] if 'max_depth' in options else 4
        max_depth = int(max_depth)
        assert max_depth > 0
        self.__max_depth = max_depth

        # Camera metadata.
        camera_pos = options['camera_pos'] if 'camera_pos' in options else (2, -2.2, 2)
        camera_pos = np.array(camera_pos).ravel()
        assert camera_pos.size == 3
        self.__camera_pos = camera_pos

        camera_lookat = options['camera_lookat'] if 'camera_lookat' in options else (0.5, 0.5, 0.5)
        camera_lookat = np.array(camera_lookat).ravel()
        assert camera_lookat.size == 3
        self.__camera_lookat = camera_lookat

        camera_up = options['camera_up'] if 'camera_up' in options else (0, 0, 1)
        camera_up = np.array(camera_up).ravel()
        assert camera_up.size == 3
        self.__camera_up = camera_up

        fov = options['fov'] if 'fov' in options else 33
        fov = float(fov)
        assert 0 < fov < 90
        self.__fov = fov

        # Lighting.
        lightmap = options['light_map'] if 'light_map' in options else 'lightmap.exr'
        lightmap = '../asset/texture/{}'.format(lightmap)
        self.__lightmap = lightmap

        # A list of objects.
        self.__hex_objects = []
        self.__tri_objects = []
        self.__shape_objects = []


    # - tri_mesh: either an obj file name or (vertices, elements)
    # - texture_img: either a texture image name (assumed to be in asset/texture) or 'chkbd_[]_{}' where an integer in []
    #   indicates the number of grids in the checkerboard and a floating point number between 0 and 1 in {} specifies the
    #   darker color in the checkerboard.
    def add_tri_mesh (self, objFile=None, vertices=None, elements=None, transforms=None, render_edges=False, color=(.5, .5, .5), texture_img=None):
        assert (objFile is not None) or (vertices is not None and elements is not None)

        tri_num = len(self.__tri_objects)
        tri_pbrt_short_name = 'tri_{:08d}.pbrt'.format(tri_num)
        tri_pbrt_name = self.__temporary_folder / tri_pbrt_short_name

        if objFile is None:
            # Load tet mesh into obj file
            tri_obj_name = self.__temporary_folder / 'tri_{:08d}.obj'.format(tri_num)
            if render_edges:
                tet2obj_with_textures(vertices, elements, obj_file_name=tri_obj_name, pbrt_file_name=tri_pbrt_name)
            else:
                tmp_error_name = self.__temporary_folder / '.tmp.error'
                tet2obj(vertices, elements, obj_file_name=tri_obj_name)
                os.system('{} {} {} 2>{}'.format('external/pbrt-v3/build/obj2pbrt', tri_obj_name, tri_pbrt_name, tmp_error_name))
        else:
            # Load obj file.
            tri_obj_name = objFile
            tmp_error_name = self.__temporary_folder / '.tmp.error'
            os.system('{} {} {} 2>{}'.format('external/pbrt-v3/build/obj2pbrt', tri_obj_name, tri_pbrt_name, tmp_error_name))

        lines = ['AttributeBegin\n',]
        # Material.
        if isinstance(color, str):
            assert len(color) == 6
            r = int(color[:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:], 16) / 255.0
            color = (r, g, b)
        color = np.array(color).ravel()
        assert color.size == 3
        for c in color:
            assert 0 <= c <= 1
        r, g, b = color

        if render_edges:
            if texture_img is None:
                texture_img = '../asset/texture/tri_grid.png'
                # You can use the code below to create a grid.png.
                # edge_width = 4
                # img_size = 64
                # texture_img = self.__temporary_folder / 'hex_{:08d}_texture.png'.format(hex_num)
                # data = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
                # data[:edge_width, :, :] = 0
                # data[-edge_width:, :, :] = 0
                # data[:, :edge_width, :] = 0
                # data[:, -edge_width:, :] = 0
                # img = Image.fromarray(data, 'RGB')
                # img.save(texture_img)
            else:
                texture_img = '../asset/texture/{}'.format(texture_img)
            lines.append('Texture "grid" "color" "imagemap" "string filename" ["{}"]\n'.format(str(texture_img)))
            lines.append('Texture "sgrid" "color" "scale" "texture tex1" "grid" "color tex2" [{} {} {}]\n'.format(r, g, b))
            lines.append('Material "matte" "texture Kd" "sgrid"\n')
        else:
            if texture_img is None:
                lines.append('Material "plastic" "color Kd" [{} {} {}] "color Ks" [{} {} {}] "float roughness" .3\n'.format(
                    r, g, b, r, g, b))
            elif 'chkbd' in texture_img:
                _, square_num, square_color = texture_img.split('_')
                square_num = int(square_num)
                square_color = np.clip(float(square_color), 0, 1)
                lines.append('Texture "checks" "spectrum" "checkerboard"\n')
                lines.append('  "float uscale" [{:d}] "float vscale" [{:d}]\n'.format(square_num, square_num))
                lines.append('  "rgb tex1" [{:f} {:f} {:f}] "rgb tex2" [{:f} {:f} {:f}]\n'.format(
                    r, g, b,
                    square_color * r, square_color * g, square_color * b
                    ))
                lines.append('Material "matte" "texture Kd" "checks"\n')
            else:
                texture_img = 'asset/texture/{}'.format(texture_img)
                lines.append('Texture "grid" "color" "imagemap" "string filename" ["{}"]\n'.format(str(texture_img)))
                lines.append('Texture "sgrid" "color" "scale" "texture tex1" "grid" "color tex2" [{} {} {}]\n'.format(r, g, b))
                lines.append('Material "matte" "texture Kd" "sgrid"\n')

        # Transforms.
        # Flipped y because pbrt uses a left-handed system.
        lines.append('Scale 1 -1 1\n')
        if transforms is not None:
            for key, vals in reversed(transforms):
                if key == 's':
                    lines.append('Scale {:f} {:f} {:f}\n'.format(vals, vals, vals))
                elif key == 'r':
                    deg = np.rad2deg(vals[0])
                    ax = vals[1:4]
                    ax /= np.linalg.norm(ax)
                    lines.append('Rotate {:f} {:f} {:f} {:f}\n'.format(deg, ax[0], ax[1], ax[2]))
                elif key == 't':
                    lines.append('Translate {:f} {:f} {:f}\n'.format(vals[0], vals[1], vals[2]))

        # Original shape.
        with open(tri_pbrt_name, 'r') as f:
            lines += f.readlines()

        lines.append('AttributeEnd\n')

        # Write back script.
        with open(tri_pbrt_name, 'w') as f:
            for l in lines:
                f.write(l)

        self.__tri_objects.append(tri_pbrt_short_name)


    # - hex_mesh is either an obj file name or (vertices, elements)
    #
    # - transforms is a list of rotation, translation, and scaling applied to the mesh applied in the order of
    #   their occurances in transforms.
    #       transforms = [rotation, translation, scaling, ...]
    #       rotation = ('r', (radians, unit_axis.x, unit_axis.y, unit_axis.z))
    #       translation = ('t', (tx, ty, tz))
    #       scaling = ('s', s)
    #   Note that we use right-handed coordinate systems in the project but pbrt uses a left-handed system.
    #   As a result, we will take care of transforming the coordinate system in this function.
    #
    # - render_edges: if True, we will generate hex pbrt files with texture coordinates indicated by the
    #   texture_map. If False, both texture_map and texture_img will be ignored.
    #
    # - color: a 3D vector between 0 and 1 or a string of 6 letters in hex. If render_voxel_edge is False,
    #   we will use a simple material. Otherwise we will generate a texture material from color and texture_img.
    #
    # - texture_img: a file name string pointing to the texture image assumed to be in asset/texture/.
    #
    # Output: it will generate a pbrt script describing the mesh and add it to self.__hex_objects.
    def add_hex_mesh (self, objFile=None, vertices=None, elements=None, transforms=None, render_edges=False, color=(.5, .5, .5), texture_img=None):
        assert (objFile is not None) or (vertices is not None and elements is not None)

        hex_num = len(self.__hex_objects)
        hex_pbrt_short_name = 'hex_{:08d}.pbrt'.format(hex_num)
        hex_pbrt_name = self.__temporary_folder / hex_pbrt_short_name

        if objFile is None:
            # Load hex mesh into obj file
            hex_obj_name = self.__temporary_folder / 'hex_{:08d}.obj'.format(hex_num)
            if render_edges:
                hex2obj_with_textures(vertices, elements, obj_file_name=hex_obj_name, pbrt_file_name=hex_pbrt_name)
            else:
                tmp_error_name = self.__temporary_folder / '.tmp.error'
                hex2obj(vertices, elements, obj_file_name=hex_obj_name)
                os.system('{} {} {} 2>{}'.format('external/pbrt-v3/build/obj2pbrt', hex_obj_name, hex_pbrt_name, tmp_error_name))
        else:
            # Load obj file.
            hex_obj_name = objFile
            tmp_error_name = self.__temporary_folder / '.tmp.error'
            os.system('{} {} {} 2>{}'.format('external/pbrt-v3/build/obj2pbrt', hex_obj_name, hex_pbrt_name, tmp_error_name))

        lines = ['AttributeBegin\n',]
        # Material.
        if isinstance(color, str):
            assert len(color) == 6
            r = int(color[:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:], 16) / 255.0
            color = (r, g, b)
        color = np.array(color).ravel()
        assert color.size == 3
        for c in color:
            assert 0 <= c <= 1
        r, g, b = color
        if render_edges:
            if texture_img is None:
                texture_img = '../asset/texture/grid_thin.png'
                # You can use the code below to create a grid.png.
                # edge_width = 4
                # img_size = 64
                # texture_img = self.__temporary_folder / 'hex_{:08d}_texture.png'.format(hex_num)
                # data = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
                # data[:edge_width, :, :] = 0
                # data[-edge_width:, :, :] = 0
                # data[:, :edge_width, :] = 0
                # data[:, -edge_width:, :] = 0
                # img = Image.fromarray(data, 'RGB')
                # img.save(texture_img)
            else:
                texture_img = '../asset/texture/{}'.format(texture_img)
            lines.append('Texture "grid" "color" "imagemap" "string filename" ["{}"]\n'.format(str(texture_img)))
            lines.append('Texture "sgrid" "color" "scale" "texture tex1" "grid" "color tex2" [{} {} {}]\n'.format(r, g, b))
            lines.append('Material "matte" "texture Kd" "sgrid"\n')
        else:
            lines.append('Material "plastic" "color Kd" [{} {} {}] "color Ks" [{} {} {}] "float roughness" .3\n'.format(
                r, g, b, r, g, b))

        # Transforms.
        # Flipped y because pbrt uses a left-handed system.
        lines.append('Scale 1 -1 1\n')
        if transforms is not None:
            for key, vals in reversed(transforms):
                if key == 's':
                    lines.append('Scale {:f} {:f} {:f}\n'.format(vals, vals, vals))
                elif key == 'r':
                    deg = np.rad2deg(vals[0])
                    ax = vals[1:4]
                    ax /= np.linalg.norm(ax)
                    lines.append('Rotate {:f} {:f} {:f} {:f}\n'.format(deg, ax[0], ax[1], ax[2]))
                elif key == 't':
                    lines.append('Translate {:f} {:f} {:f}\n'.format(vals[0], vals[1], vals[2]))

        # Original shape.
        with open(hex_pbrt_name, 'r') as f:
            lines += f.readlines()

        lines.append('AttributeEnd\n')

        # Write back script.
        with open(hex_pbrt_name, 'w') as f:
            for l in lines:
                f.write(l)

        self.__hex_objects.append(hex_pbrt_short_name)

    # - shape_info: a dictionary.
    # Examples for running this:
    #  shape_info = {'name': 'sphere', 'center': [0., 0., 0.], 'radius': 0.01}
    def add_shape_mesh (self, shape_info, transforms=None, color=(.5, .5, .5)):
        shape_num = len(self.__shape_objects)
        shape_pbrt_short_name = 'shape_{:08d}.pbrt'.format(shape_num)
        shape_pbrt_name = self.__temporary_folder / shape_pbrt_short_name

        lines = ['AttributeBegin\n',]
        # Material.
        if isinstance(color, str):
            assert len(color) == 6
            r = int(color[:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:], 16) / 255.0
            color = (r, g, b)
        color = np.array(color).ravel()
        assert color.size == 3
        for c in color:
            assert 0 <= c <= 1
        r, g, b = color
        lines.append('Material "plastic" "color Kd" [{} {} {}] "color Ks" [{} {} {}] "float roughness" .3\n'.format(
            r, g, b, r, g, b))
 
        # Transforms.
        # Flipped y because pbrt uses a left-handed system.
        lines.append('Scale 1 -1 1\n')
        if transforms is not None:
            for key, vals in reversed(transforms):
                if key == 's':
                    lines.append('Scale {:f} {:f} {:f}\n'.format(vals, vals, vals))
                elif key == 'r':
                    deg = np.rad2deg(vals[0])
                    ax = vals[1:4]
                    ax /= np.linalg.norm(ax)
                    lines.append('Rotate {:f} {:f} {:f} {:f}\n'.format(deg, ax[0], ax[1], ax[2]))
                elif key == 't':
                    lines.append('Translate {:f} {:f} {:f}\n'.format(vals[0], vals[1], vals[2]))

        # Original shape.
        shape_name = shape_info['name']
        if shape_name == 'curve':
            points = np.array(shape_info['point']).ravel()
            assert points.size == 12
            type_info = '"string type" "flat"'
            if 'type' in shape_info:
                type_info = '"string type" "{}"'.format(shape_info['type'])
            width_info = '"float width" [1.0]'
            if 'width' in shape_info:
                width_info = '"float width" [{}]'.format(float(shape_info['width']))
            lines.append('Shape "curve" "point P" [' + ' '.join([str(v) for v in points])
                + '] {} {}\n'.format(type_info, width_info))
        elif shape_name == 'sphere':
            radius = float(shape_info['radius'])
            center = np.array(shape_info['center']).ravel()
            assert center.size == 3
            lines.append('Translate {:f} {:f} {:f}\n'.format(center[0], center[1], center[2]))
            lines.append('Shape "sphere" "float radius" [{:f}]'.format(radius))
        elif shape_name == 'cylinder':
            radius = float(shape_info['radius'])
            zmin = float(shape_info['zmin'])
            zmax = float(shape_info['zmax'])
            lines.append('Shape "cylinder" "float radius" [{:f}] "float zmin" [{:f}] "float zmax" [{:f}]'.format(
                radius, zmin, zmax))
        else:
            raise NotImplementedError

        lines.append('AttributeEnd\n')

        # Write back script.
        with open(shape_pbrt_name, 'w') as f:
            for l in lines:
                f.write(l)

        self.__shape_objects.append(shape_pbrt_short_name)


    # Call this function after you have set up add_hex_mesh and add_tri_mesh.
    def render(self, verbose=False, light_rgb=(1., 1., 1.), nproc=None):
        scene_pbrt_name = self.__temporary_folder / 'scene.pbrt'
        with open(scene_pbrt_name, 'w') as f:
            x_res, y_res = self.__resolution
            f.write('Film "image" "integer xresolution" [{:d}] "integer yresolution" [{:d}]\n'.format(x_res, y_res))
            f.write('    "string filename" "{:s}.exr"\n'.format(self.__file_name_only))

            f.write('\n')
            f.write('Sampler "halton" "integer pixelsamples" [{:d}]\n'.format(self.__sample))
            f.write('Integrator "path" "integer maxdepth" {:d}\n'.format(self.__max_depth))

            f.write('\n')
            # Flipped y because pbrt uses a left-handed coordinate system.
            cpx, cpy, cpz = self.__camera_pos
            clx, cly, clz = self.__camera_lookat
            cux, cuy, cuz = self.__camera_up
            f.write('LookAt {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(
                cpx, -cpy, cpz,
                clx, -cly, clz,
                cux, -cuy, cuz))
            f.write('Camera "perspective" "float fov" [{:f}]\n'.format(self.__fov))

            f.write('\n')
            f.write('WorldBegin\n')

            f.write('\n')
            f.write('AttributeBegin\n')
            f.write('LightSource "infinite" "string mapname" "{}" "color scale" [{:f}, {:f}, {:f}]\n'.format(str(
                self.__lightmap), light_rgb[0], light_rgb[1], light_rgb[2]))
            f.write('AttributeEnd\n')

            f.write('\n')
            for hex_pbrt_name in self.__hex_objects:
                f.write('Include "{}"\n'.format(hex_pbrt_name))

            for tri_pbrt_name in self.__tri_objects:
                f.write('Include "{}"\n'.format(tri_pbrt_name))

            for shape_pbrt_name in self.__shape_objects:
                f.write('Include "{}"\n'.format(shape_pbrt_name))

            f.write('\n')
            f.write('WorldEnd\n')

        verbose_flag = ' ' if verbose else '--quiet'
        thread_flag = ' ' if nproc is None else '--nthreads {:d}'.format(int(nproc))
        os.system('{} {} {} {}'.format('external/pbrt-v3/build/pbrt',
            verbose_flag, thread_flag, scene_pbrt_name))
        os.system('convert {}.exr {}.png'.format(self.__file_name_only, self.__file_name_only))

        os.remove('{}.exr'.format(self.__file_name_only))

        # Cleanup data.
        shutil.rmtree(self.__temporary_folder)




# Filter unreferenced vertices in a mesh.
# Input:
# - vertices: n x l.
# - elements: m x k.
# This function checks all rows in elements and generates a new (vertices, elements) pair such that all
# vertices are referenced.
def filter_unused_vertices(vertices, elements):
    vert_num = vertices.shape[0]
    elem_num = elements.shape[0]

    used = np.zeros(vert_num)
    for e in elements:
        for ei in e:
            used[ei] = 1

    remap = np.ones(vert_num) * -1
    used_so_far = 0
    for idx, val in enumerate(used):
        if val > 0:
            remap[idx] = used_so_far
            used_so_far += 1

    new_vertices = []
    for idx, val in enumerate(used):
        if val > 0:
            new_vertices.append(vertices[idx])
    new_vertices = np.array(new_vertices)

    new_elements = []
    for e in elements:
        new_ei = [remap[ei] for ei in e]
        new_elements.append(new_ei)
    new_elements = np.array(new_elements).astype(int)
    return new_vertices, new_elements



def fix_tet_faces(verts):
    verts = np.array(verts)
    v0, v1, v2, v3 = verts
    f = []
    if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
        f = [
            (0, 1, 2),
            (2, 1, 3),
            (1, 0, 3),
            (0, 2, 3),
        ]
    else:
        f = [
            (1, 0, 2),
            (1, 2, 3),
            (0, 1, 3),
            (2, 0, 3),
        ]

    return np.array(f).astype(int)


### TETRAHEDRAL MESH ###
# Given a tet mesh, save it as an obj file with texture coordinates.
def tet2obj_with_textures (vertices, elements, obj_file_name=None, pbrt_file_name=None):
    v, f = tet2obj(vertices, elements)

    v_out = []
    f_out = []
    v_cnt = 0
    for fi in f:
        fi_out = [v_cnt, v_cnt + 1, v_cnt + 2]
        f_out.append(fi_out)
        v_cnt += 3
        for vi in fi:
            v_out.append(np.array(v[vi]))

    texture_map = [[0, 0], [1, 0], [0, 1]]
    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v_out:
                f_obj.write('v {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            for u, v in texture_map:
                f_obj.write('vt {:6f} {:6f}\n'.format(u, v))
            for ff in f_out:
                f_obj.write('f {:d}/1 {:d}/2 {:d}/3\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))

    if pbrt_file_name is not None:
        with open(pbrt_file_name, 'w') as f_pbrt:
            f_pbrt.write('AttributeBegin\n')
            f_pbrt.write('Shape "trianglemesh"\n')

            # Log point data.
            f_pbrt.write('  "point3 P" [\n')
            for vv in v_out:
                f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            f_pbrt.write(']\n')

            # Log texture data.
            f_pbrt.write('  "float uv" [\n')
            for _ in range(int(len(v_out) / 3)):
                f_pbrt.write('  0 0\n')
                f_pbrt.write('  1 0\n')
                f_pbrt.write('  0 1\n')
            f_pbrt.write(']\n')

            # Log face data.
            f_pbrt.write('  "integer indices" [\n')
            for ff in f_out:
                f_pbrt.write('  {:d} {:d} {:d}\n'.format(ff[0], ff[1], ff[2]))
            f_pbrt.write(']\n')
            f_pbrt.write('AttributeEnd\n')

# Given tet_mesh, return vert and elements that describes the surface mesh as a triangle mesh, and optionally stores an obj file.
# Output:
# - vertices: an n x 3 double array.
# - elements: an m x 4 integer array.
def tet2obj (vertices, elements, obj_file_name=None):
    vertex_num = vertices.shape[0]
    element_num = elements.shape[0]

    v = vertices.copy()

    face_dict = {}
    for i in range(element_num):
        fi = elements[i]
        element_vert = []
        for vi in fi:
            element_vert.append(vertices[vi])
        element_vert = np.array(element_vert)
        face_idx = fix_tet_faces(element_vert)
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = np.array(f).astype(int)

    v, f = filter_unused_vertices(v, f)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            for ff in f:
                f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))

    return v, f


### HEXAHEDRAL MESH ###
# Given a hex mesh, save it as an obj file with texture coordinates.
def hex2obj_with_textures (vertices, elements, obj_file_name=None, pbrt_file_name=None, texture_map=None):
    v, f = hex2obj(vertices, elements)

    v_out = []
    f_out = []
    v_cnt = 0
    for fi in f:
        fi_out = [v_cnt, v_cnt + 1, v_cnt + 2, v_cnt + 3]
        f_out.append(fi_out)
        v_cnt += 4
        for vi in fi:
            v_out.append(np.array(v[vi]))

    if texture_map is None:
        texture_map = [[0, 0], [1, 0], [1, 1], [0, 1]]
    texture_map = np.array(texture_map)
    assert texture_map.shape == (4, 2)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v_out:
                f_obj.write('v {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            for u, v in texture_map:
                f_obj.write('vt {:6f} {:6f}\n'.format(u, v))
            for ff in f_out:
                f_obj.write('f {:d}/1 {:d}/2 {:d}/3\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
                f_obj.write('f {:d}/1 {:d}/3 {:d}/4\n'.format(ff[0] + 1, ff[2] + 1, ff[3] + 1))

    if pbrt_file_name is not None:
        with open(pbrt_file_name, 'w') as f_pbrt:
            f_pbrt.write('AttributeBegin\n')
            f_pbrt.write('Shape "trianglemesh"\n')

            # Log point data.
            f_pbrt.write('  "point3 P" [\n')
            for vv in v_out:
                f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            f_pbrt.write(']\n')

            # Log texture data.
            f_pbrt.write('  "float uv" [\n')
            for _ in range(int(len(v_out) / 4)):
                f_pbrt.write('  0 0\n')
                f_pbrt.write('  1 0\n')
                f_pbrt.write('  1 1\n')
                f_pbrt.write('  0 1\n')
            f_pbrt.write(']\n')

            # Log face data.
            f_pbrt.write('  "integer indices" [\n')
            for ff in f_out:
                # f_pbrt.write('  {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(ff[0], ff[1], ff[2], ff[0], ff[2], ff[3]))
                f_pbrt.write('  {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(ff[0], ff[3], ff[2], ff[0], ff[2], ff[1]))
            f_pbrt.write(']\n')
            f_pbrt.write('AttributeEnd\n')

# Given a hex mesh, return vert and elements that describes the surface mesh as a quad mesh, and optionally stores an obj file.
# - vertices: an n x 3 double array.
# - faces: an m x 4 integer array.
def hex2obj (vertices, elements, obj_file_name=None, obj_type='quad'):
    vertex_num = vertices.shape[0]
    element_num = elements.shape[0]

    v = vertices.copy()

    face_dict = {}
    face_idx = [
        # (0, 1, 2, 3),
        # (4, 5, 6, 7),
        # (0, 4, 5, 1),
        # (2, 3, 7, 6),
        # (1, 5, 6, 2),
        (0, 3, 7, 4)
    ]
    for i in range(element_num):
        fi = elements[i]
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = np.array(f).astype(int)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            if obj_type == 'quad':
                for ff in f:
                    f_obj.write('f {} {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1, ff[3] + 1))
            elif obj_type == 'tri':
                for ff in f:
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[2] + 1, ff[3] + 1))
            else:
                raise NotImplementedError

    return v, f



# Input:
# - vertices and elements
# - active_elements: a list of elements that you would like to keep.
# Output:
# - vertices and elements 
def filter_elements (vertices, elements, active_elements):
    vertex_num = vertices.shape[0]
    element_num = elements.shape[0]
    vertex_indices = np.zeros(vertex_num)
    for e_idx in active_elements:
        vertex_indices[elements[e_idx]] = 1
    remap = -np.ones(vertex_num)
    cnt = 0

    # Vertices
    vs = []
    for i in range(vertex_num):
        if vertex_indices[i] == 1:
            remap[i] = cnt
            cnt += 1
            vs.append(vertices[i])
    vs = np.array(vs)

    # Elements
    es = []
    for e_idx in active_elements:
        es.append([remap[ei] for ei in elements[e_idx]])
    es = np.array(es).astype(int)

    return vs, es




def export_mp4(folder_name, mp4_name, fps, name_prefix=''):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith('.png')]
    frame_names = sorted(frame_names)

    # Create a temporary folder.
    tmp_folder = Path('_export_mp4')
    create_folder(tmp_folder, exist_ok=False)
    for i, f in enumerate(frame_names):
        shutil.copyfile(f, tmp_folder / '{:08d}.png'.format(i))
        
    os.system(f"ffmpeg -r {fps} -i '{tmp_folder}/%08d.png' -vcodec libx264 -y {mp4_name}")

    # Delete temporary folder.
    shutil.rmtree(tmp_folder)