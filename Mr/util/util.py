from __future__ import print_function
import torch
import numpy as np
import os
import uuid
import glob

# from options.base_options import MANIFOLD_DIR
MANIFOLD_DIR = r'/home/amax/Manifold-master/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


MESH_EXTENSIONS = [
    '.obj',
]


def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


# def pad(input_arr, target_length, val=0, dim=1):
#     shp = input_arr.shape
#     npad = [(0, 0) for _ in range(len(shp))]
#     npad[dim] = (0, target_length - shp[dim])
#     return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

# def seg_accuracy(predicted, ssegs, meshes):
#     correct = 0
#     ssegs = ssegs.squeeze(-1)
#     correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
#     for mesh_id, mesh in enumerate(meshes):
#         correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
#         edge_areas = torch.from_numpy(mesh.get_edge_areas())
#         correct += (correct_vec.float() * edge_areas).sum()
#     return correct

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        net work
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


# def get_heatmap_color(value, minimum=0, maximum=1):
#     minimum, maximum = float(minimum), float(maximum)
#     ratio = 2 * (value-minimum) / (maximum - minimum)
#     b = int(max(0, 255*(1 - ratio)))
#     r = int(max(0, 255*(ratio - 1)))
#     g = 255 - b - r
#     return r, g, b


# def normalize_np_array(np_array):
#     min_value = np.min(np_array)
#     max_value = np.max(np_array)
#     return (np_array - min_value) / (max_value - min_value)


# def calculate_entropy(np_array):
#     entropy = 0
#     np_array /= np.sum(np_array)
#     for a in np_array:
#         if a != 0:
#             entropy -= a * np.log(a)
#     entropy /= np.log(np_array.shape[0])
#     return entropy

################################point2mesh


def manifold_upsample(fname, save_path, Mesh, num_faces=2000, res=3000, simplify=True):
    # export before upsample
    # fname = os.path.join(save_path, 'recon_{}.obj'.format('61994'))
    # mesh.export(fname)

    # temp_file = os.path.join(save_path, random_file_name('obj'))
    optss = ' ' + str(res) if res is not None else ''

    manifold_script_path = os.path.join(MANIFOLD_DIR, 'manifold')
    print(manifold_script_path)
    if not os.path.exists(manifold_script_path):
        raise FileNotFoundError(f'{manifold_script_path} not found')
    cmd = "{} {} {}".format(manifold_script_path, fname, save_path + optss)
    os.system(cmd)
    # print('a',a)

    if simplify:
        cmd = "{} -i {} -o {} -f {}".format(os.path.join(MANIFOLD_DIR, 'simplify'), save_path, save_path, num_faces)
        os.system(cmd)
        # print('b',b)

    m_out = Mesh(save_path, hold_history=True, device=torch.device('cpu'))
    # m_out = load_obj(save_path)
    # fname = os.path.join(save_path, 'recon_{}_after.obj'.format(len(m_out.faces)))
    # m_out.export(fname)
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(save_path)[0] + '*'))]
    return m_out


def read_pts(pts_file):
    '''
    :param pts_file: file path of a plain text list of points
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    which is typical for (plaintext) .ply or .xyz
    :return: xyz, normals
    '''
    xyz, normals = [], []
    with open(pts_file, 'r') as f:
        # line = f.readline()
        spt = f.read().split('\n')
        # while line:
        for line in spt:
            parts = line.strip().split(' ')
            try:
                x = np.array(parts, dtype=np.float32)
                xyz.append(x[:3])
                normals.append(x[3:])
            except:
                pass
    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces
    # vs, faces = [], []
    # f = open(file)
    # for line in f:
    #     line = line.strip()
    #     splitted_line = line.split()
    #     if not splitted_line:
    #         continue
    #     elif splitted_line[0] == 'v':
    #         vs.append([float(v) for v in splitted_line[1:4]])
    #     elif splitted_line[0] == 'f':
    #         face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
    #         assert len(face_vertex_ids) == 3
    #         face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
    #                            for ind in face_vertex_ids]
    #         faces.append(face_vertex_ids)
    # f.close()
    # vs = np.asarray(vs)
    # faces = np.asarray(faces, dtype=int)
    # assert np.logical_and(faces >= 0, faces < len(vs)).all()
    # return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'


def get_num_parts(num_faces):
    lookup_num_parts = [1, 2, 4, 8]
    # num_default = [8000, 16000, 20000]
    num_default = [800, 1600, 2000]
    num_parts = lookup_num_parts[np.digitize(num_faces, num_default, right=True)]
    # num_parts = 2
    return num_parts


def get_num_samples(cur_iter):
    samples = 25000
    begin_samples = 15000
    upsamp = 1000
    slope = (samples - begin_samples) / int(0.8 * upsamp)
    return int(slope * min(cur_iter, 0.8 * upsamp)) + begin_samples


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def sample_surface(faces, vs, count):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices
    faces: triangle faces (torch.long)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    normals: (count, 3) corresponding face normals for points
    """
    bsize, nvs, _ = vs.shape
    weights, normal = face_areas_normals(faces, vs)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)

    return samples, normals


def face_areas_normals(faces, vs):
    # norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tris = vertices[faces]
    # n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    face_normals = torch.cross(vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
                               vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def local_nonuniform_penalty(mesh):
    # non-uniform penalty
    area = mesh_area(mesh)
    diff = area[mesh.gfmm][:, 0:1] - area[mesh.gfmm][:, 1:]
    penalty = torch.norm(diff, dim=1, p=1)
    loss = penalty.sum() / penalty.numel()
    return loss


def cal_edge_loss(mesh):
    coord = mesh.vs
    edges = torch.tensor(mesh.edges, dtype=torch.long, device=torch.device('cuda'))
    edges_flatten = edges.reshape(-1)
    coord_flatten = coord[edges_flatten, :]
    edge_coords = coord_flatten.reshape((-1, 2, 3))
    edge_length = torch.mean(torch.sum((edge_coords[:, 0] - edge_coords[:, 1]) ** 2, dim=1))
    return edge_length


def mesh_area(mesh):
    vs = mesh.vs
    faces = mesh.faces
    v1 = vs[faces[:, 1]] - vs[faces[:, 0]]
    v2 = vs[faces[:, 2]] - vs[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area
