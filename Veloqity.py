import subprocess
import sys
from threading import Thread
import threading
import time
import math
import json


def travel_dist(a, b):
    if abs(a[1]-b[1])+abs(a[0]-b[0]) == 1:
        return 0.0001
    m = max(abs(a[1]-b[1]), abs(a[0]-b[0]))
    if a[1] != b[1] and a[0] != b[0]:
        m += 3
    return m*3+math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)/20
 
def findOrder(n, m):
    result=[0]*(n*m)
    result[0] = (0, 0)
    k=1
    i=j=0
    while(k<n*m):
        while i>=1 and j<n-1:
            i-=1
            j+=1
            result[k] = (i, j)
            k+=1
        if j<n-1:
            j+=1
            result[k] = (i, j)
            k+=1
        elif i<m-1:
            i+=1
            result[k] = (i, j)
            k+=1
        while i<m-1 and j>=1:
            i+=1
            j-=1
            result[k] = (i, j)
            k+=1
        if i<m-1:
            i+=1
            result[k] = (i, j)
            k+=1
        elif j<n-1:
            j+=1
            result[k] = (i, j)
            k+=1
    return result


DEFAULT_MODEL = 'duck.stl'

with open('vert_shader.glsl', 'r') as f:
    VERTEX_SHADER = f.read()


with open('frag_shader.glsl', 'r') as f:
    FRAGMENT_SHADER = f.read()


def install(library):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library])

missing_libs = []
clr_init = False

try:
    import colorama
    clr = colorama
    clr.init()
    clr_init = True
except ImportError:
    missing_libs.append(('Colorama', 'colorama'))


if clr_init:
    def error(message):
        print(clr.Fore.RED + clr.Style.BRIGHT +
            'Error: ' + clr.Style.RESET_ALL + message)
        sys.exit(1)


    def info(message):
        print(clr.Fore.YELLOW + clr.Style.BRIGHT +
            'Info: ' + clr.Style.RESET_ALL + message)
else:
    def error(message):
        print('Error: ' + message)
        sys.exit(1)


    def info(message):
        print('Info: ' + message)

try:
    import trimesh
    from trimesh.voxel.creation import voxelize
except ImportError:
    missing_libs.append(('Trimesh', 'trimesh[easy]'))

try:
    import customtkinter
except ImportError:
    missing_libs.append(('CustomTkinter', 'customtkinter'))

try:
    from OpenGL import GL
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    from OpenGL.GLUT import *
except ImportError:
    missing_libs.append(('PyOpenGL', 'PyOpenGL'))

try:
    from pyopengltk import OpenGLFrame
except ImportError:
    missing_libs.append(('PyOpenGLTk', 'pyopengltk'))

try:
    import numpy
except ImportError:
    missing_libs.append(('Numpy', 'numpy'))

try:
    import pants
except ImportError:
    missing_libs.append(('Ant Colony Optimization Meta-Heuristic', 'ACO-Pants'))

if len(missing_libs) > 0:
    for lib in missing_libs:
        info('Missing library: ' + lib[0])
    print('Would you like to install the missing libraries automatically?')
    ans = input('Y/N: ')
    if ans.lower() == 'y':
        for lib in missing_libs:
            info('Installing ' + lib[0])
            install(lib[1])
        subprocess.check_call([sys.executable, __file__])
    else:
        info('To install the missing libraries manually, run the following commands:')
        for lib in missing_libs:
            print(sys.executable.split('\\')[-1].split('/')[-1]+' -m pip install '+lib[1])
    sys.exit(0)

ctk = customtkinter
mesh = None
voxels = None
working_mesh = None
np = numpy
scale = 5
cam_pitch = 0
cam_yaw = 0
APPR_SPEED = 0.05

def warp_encode_cmd_to_num(x, y, z, action):
    return (str(x).zfill(2))+(str(y).zfill(2))+(str(z).zfill(2))+(str(action).zfill(2))

def legacy_encode_cmd_to_num(x, y, z, action):
    num = z
    num += y*16
    num += x*256
    num += action*4096
    return num

def legacy(voxels=None, printer=None, givename=False):
    if givename:
        return 'Legacy'
    if printer['type'] != '3axis':
        return 'Printer type not supported'
    if len(printer['extruders']) < 1:
        return 'Printer has no extruders'
    if 'block' not in printer['extruders']:
        return 'Printer does not have a block extruder'
    if 'support' not in printer['extruders']:
        return 'Printer does not have a support extruder'
    if printer['build_volume'][0] != 16 or printer['build_volume'][1] != 16 or printer['build_volume'][2] != 16:
        return 'Printer build volume is not 16x16x16; legacy slicer only supports such build volume'
    shape = []
    data = []
    for x in range(16):
        shape.append([])
        for y in range(16):
            shape[-1].append(['-']*16)
    #find offset to center model
    v_shape = voxels.shape
    x_offset = math.floor((16-v_shape[0])/2)
    y_offset = math.floor((16-v_shape[1])/2)
    z_offset = math.floor((16-v_shape[2])/2)
    vs = voxels.points_to_indices(voxels.points)
    for v in vs:
        shape[v[0]+x_offset][v[2]+z_offset][v[1]+1] = 'b'
    for x in range(16):
        for y in range(16):
            for z in range(16):
                if shape[x][y][z] != '-':
                    continue
                if z == 15:
                    continue
                for over_z in range(z+1, 16):
                    if shape[x][y][over_z] != '-':
                        shape[x][y][z] = 's'
                        break
    for z in range(16):
        for y in range(16):
            line = ''
            for x in range(16):
                line += shape[x][y][z]
        #     print(line)
        # print('----------------------------------------------------------------')
    
    data.append(legacy_encode_cmd_to_num(0, 0, 0, 2))
    print_head_x = 0
    print_head_y = 0
    for z in range(16):
        zigzag = list(findOrder(16, 16))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 's':
                data.append(legacy_encode_cmd_to_num(pos[0], pos[1], z, 1))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 'b':
                data.append(legacy_encode_cmd_to_num(pos[0], pos[1], z, 0))
    data.append(legacy_encode_cmd_to_num(0, 0, 15, 3))
    return [data]

def warp_slicer(voxels=None, printer=None, givename=False):
    if givename:
        return 'Warp Slicer'
    if printer['type'] != '3axis':
        return 'Printer type not supported'
    if len(printer['extruders']) < 1:
        return 'Printer has no extruders'
    shape = []
    data = []
    for x in range(64):
        shape.append([])
        for y in range(64):
            shape[-1].append(['-']*64)
    v_shape = voxels.shape
    x_offset = math.floor((64-v_shape[0])/2)
    y_offset = math.floor((64-v_shape[1])/2)
    z_offset = math.floor((64-v_shape[2])/2)
    vs = voxels.points_to_indices(voxels.points)
    for v in vs:
        shape[v[0]+x_offset][v[2]+z_offset][v[1]+1] = 'b'
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if shape[x][y][z] != '-':
                    continue
                if z == 15:
                    continue
                for over_z in range(z+1, 64):
                    if shape[x][y][over_z] != '-':
                        shape[x][y][z] = 's'
                        break
    for z in range(64):
        for y in range(64):
            line = ''
            for x in range(64):
                line += shape[x][y][z]
        #     print(line)
        # print('----------------------------------------------------------------')
    for z in range(16):
        zigzag = list(findOrder(64, 64))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 's':
                data.append(warp_encode_cmd_to_num(pos[0], z, pos[1], 7))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 'b':
                data.append(warp_encode_cmd_to_num(pos[0], z, pos[1], 1))
    data.append(warp_encode_cmd_to_num(0, 0, 15, 99))
    return data

def tower_encode_cmd_to_num(x, y, z, action):
    num = x
    num += y*256
    num += z*256*256
    num += action*256*256*256
    return num

def tower(voxels=None, printer=None, givename=False):
    if givename:
        return 'Tower'
    if printer['type'] != '3axis':
        return 'Printer type not supported'
    if len(printer['extruders']) < 1:
        return 'Printer has no extruders'
    data = []
    shape = []
    for x in range(256):
        print(x)
        shape.append([])
        for y in range(256):
            shape[-1].append(['-']*256)
    #find offset to center model
    v_shape = voxels.shape
    x_offset = math.floor((256-v_shape[0])/2)
    y_offset = math.floor((256-v_shape[1])/2)
    z_offset = math.floor((256-v_shape[2])/2)
    vs = voxels.points_to_indices(voxels.points)
    highest_z = 0
    min_x = 256
    min_y = 256
    max_x = 0
    max_y = 0
    for v in vs:
        shape[v[0]+x_offset][v[2]+z_offset][v[1]+1] = 'b'
        highest_z = max(highest_z, v[1]+1)
        min_x = min(min_x, v[0]+x_offset)
        min_y = min(min_y, v[2]+z_offset)
        max_x = max(max_x, v[0]+x_offset)
        max_y = max(max_y, v[2]+z_offset)
    for x in range(256):
        if x < min_x or x > max_x:
            continue
        print(x)
        for y in range(256):
            if y < min_y or y > max_y:
                continue
            for z in range(256):
                if z > highest_z:
                    continue
                if shape[x][y][z] != '-':
                    continue
                if z == 255:
                    continue
                for over_z in range(z+1, 256):
                    if shape[x][y][over_z] != '-':
                        shape[x][y][z] = 's'
                        break
    # for z in range(256):
    #     for y in range(256):
    #         line = ''
    #         for x in range(256):
    #             line += shape[x][y][z]
    #         print(line)
    #     print('----------------------------------------------------------------')
    for z in range(256):
        print(z)
        if z > highest_z:
            break
        zigzag = list(findOrder(256, 256))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 's':
                data.append(tower_encode_cmd_to_num(pos[0], pos[1], z, 0))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 'b':
                data.append(tower_encode_cmd_to_num(pos[0], pos[1], z, 1))
    data.append(tower_encode_cmd_to_num(0, 0, 0, 3))
    return data
    
    
def cartesian_standard(voxels=None, printer=None, givename=False):
    if givename:
        return 'Cartesian Standard'
    if printer['type'] != '3axis':
        return 'Printer type not supported'
    if len(printer['extruders']) < 1:
        return 'Printer has no extruders'
    if 'block' not in printer['extruders']:
        return 'Printer does not have a block extruder'
    if 'support' not in printer['extruders']:
        return 'Printer does not have a support extruder'
    shape = []
    data = []
    for x in range(printer['build_volume'][0]):
        shape.append([])
        for y in range(printer['build_volume'][2]):
            shape[-1].append(['-']*printer['build_volume'][1])
    #find offset to center model
    v_shape = voxels.shape
    x_offset = math.floor((printer['build_volume'][0]-v_shape[0])/2)
    z_offset = math.floor((printer['build_volume'][2]-v_shape[2])/2)
    vs = voxels.points_to_indices(voxels.points)
    for v in vs:
        shape[v[0]+x_offset][v[2]+z_offset][v[1]+1] = 'b'
    for x in range(printer['build_volume'][0]):
        for y in range(printer['build_volume'][2]):
            for z in range(printer['build_volume'][1]):
                if shape[x][y][z] != '-':
                    continue
                if z == printer['build_volume'][1]-1:
                    continue
                for over_z in range(z+1, printer['build_volume'][1]):
                    if shape[x][y][over_z] != '-':
                        shape[x][y][z] = 's'
                        break
                    
    for z in range(printer['build_volume'][1]):
        for y in range(printer['build_volume'][2]):
            line = ''
            for x in range(printer['build_volume'][0]):
                line += shape[x][y][z]
            print(line)
        # input()
        # print('----------------------------------------------------------------')
    for z in range(printer['build_volume'][1]):
        print(z)
        zigzag = list(findOrder(printer['build_volume'][0], printer['build_volume'][2]))
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 's':
                data.extend([7, pos[0], pos[1], z, -1])
        for pos in zigzag:
            if shape[pos[0]][pos[1]][z] == 'b':
                data.extend([0, pos[0], pos[1], z, -1])
    data.append(-1)
    return [data]

def place_5axis(type, block_x, block_y, block_z, supported_from):
    out = [type]
    sup_vec = [0, 0, 0]
    if supported_from == 'bottom':
        sup_vec = [0.5, 0.5, 0]
    elif supported_from == '-x':
        sup_vec = [0, 0.5, 0.5]
    elif supported_from == '+x':
        sup_vec = [1, 0.5, 0.5]
    elif supported_from == '-y':
        sup_vec = [0.5, 0, 0.5]
    elif supported_from == '+y':
        sup_vec = [0.5, 1, 0.5]
    out.extend([block_x+(sup_vec[0]-0.5)*-12, block_y+(sup_vec[1]-0.5)*-10, block_z+8-sup_vec[2]*10])
    out.extend([block_x+sup_vec[0], block_y+sup_vec[1], block_z+sup_vec[2]])
    out.append(-1)
    return out
def drone_5axis(voxels=None, printer=None, givename=False):
    if givename:
        return 'Drone 5-Axis'
    if printer['type'] != '5axis':
        return 'Printer type not supported'
    BLOCK = 0
    SUPPORT = 7
    data = []
    shape = []
    v_shape = voxels.shape
    b_xsize = v_shape[0]+10
    b_ysize = v_shape[2]+10
    b_zsize = v_shape[1]+1
    for x in range(b_xsize):
        shape.append([])
        for y in range(b_ysize):
            shape[-1].append(['-']*(b_zsize))
    vs = voxels.points_to_indices(voxels.points)
    for v in vs:
        shape[v[0]+10][v[2]+10][v[1]+1] = 'u'
    for z in range(b_zsize):
        for y in range(b_ysize):
            line = ''
            for x in range(b_xsize):
                line += shape[x][y][z]
            print(line)
        print('----------------------------------------------------------------')
    for z in range(1, b_zsize):
        nextinline = []
        cnt_to_check = 0
        for y in range(b_ysize):
            for x in range(b_xsize):
                if shape[x][y][z] == 'u':
                    cnt_to_check += 1
                    if shape[x][y][z-1] == 'b':
                        shape[x][y][z] = 'g'
                        cnt_to_check += 1
                    else:
                        print('found', x, y, z)
        print('cnt', cnt_to_check)
        while cnt_to_check > 0:
            if len(nextinline) == 0:
                for y in range(b_ysize):
                    for x in range(b_xsize):
                        if shape[x][y][z] == 'g':
                            nextinline.append((x, y))
                            break
                    if len(nextinline) != 0:
                        break
            if len(nextinline) == 0:
                for y in range(b_ysize):
                    for x in range(b_xsize):
                        if shape[x][y][z] == 'u':
                            nextinline.append((x, y))
                            for zr in range(z-1, -1, -1):
                                if shape[x][y][zr] == '-':
                                    shape[x][y][zr] = 's'
                                else:
                                    break
                            break
                    if len(nextinline) != 0:
                        break

            pos_to_check = nextinline.pop(0)
            print(pos_to_check)
            x = pos_to_check[0]
            y = pos_to_check[1]
            print(shape[x][y][z])
            if shape[x][y][z] in 'ug':
                shape[x][y][z] = 'b'
                print('eeeee')
                if x > 0:
                    print('x-1')
                    if shape[x-1][y][z] in 'ug' and (x-1, y) not in nextinline:
                        nextinline.append((x-1, y))
                if x < b_xsize-1:
                    print('x+1')
                    if shape[x+1][y][z] in 'ug' and (x+1, y) not in nextinline:
                        nextinline.append((x+1, y))
                if y > 0:
                    print('y-1')
                    if shape[x][y-1][z] in 'ug' and (x, y-1) not in nextinline:
                        nextinline.append((x, y-1))
                if y < b_ysize-1:
                    print('y+1')
                    if shape[x][y+1][z] in 'ug' and (x, y+1) not in nextinline:
                        nextinline.append((x, y+1))
            print(nextinline)
            cnt_to_check = 0
            for y in range(b_ysize):
                for x in range(b_xsize):
                    if shape[x][y][z] in 'ug':
                        cnt_to_check += 1
    for z in range(b_zsize):
        for y in range(b_ysize):
            line = ''
            for x in range(b_xsize):
                line += shape[x][y][z]
            print(line)
        print('----------------------------------------------------------------')
    data = []
    for z in range(b_zsize):
        print(z)
        cnt_to_place = 0
        for y in range(b_ysize):
            for x in range(b_xsize):
                if shape[x][y][z] in 'bs':
                    cnt_to_place += 1
        while cnt_to_place > 0:
            print(cnt_to_place)
            for y in range(b_ysize):
                for x in range(b_xsize):
                    if shape[x][y][z] in '-p':
                        continue
                    if z == 0 or shape[x][y][z-1] != '-':
                        data.extend(place_5axis({'b': BLOCK, 's': SUPPORT}[shape[x][y][z]], x, y, z, 'bottom'))
                        shape[x][y][z] = 'p'
                    elif x > 0 and shape[x-1][y][z] == 'p':
                        data.extend(place_5axis({'b': BLOCK, 's': SUPPORT}[shape[x][y][z]], x, y, z, '-x'))
                        shape[x][y][z] = 'p'
                    elif x < b_xsize-1 and shape[x+1][y][z] == 'p':
                        data.extend(place_5axis({'b': BLOCK, 's': SUPPORT}[shape[x][y][z]], x, y, z, '+x'))
                        shape[x][y][z] = 'p'
                    elif y > 0 and shape[x][y-1][z] == 'p':
                        data.extend(place_5axis({'b': BLOCK, 's': SUPPORT}[shape[x][y][z]], x, y, z, '-y'))
                        shape[x][y][z] = 'p'
                    elif y < b_ysize-1 and shape[x][y+1][z] == 'p':
                        data.extend(place_5axis({'b': BLOCK, 's': SUPPORT}[shape[x][y][z]], x, y, z, '+y'))
                        shape[x][y][z] = 'p'
            cnt_to_place = 0
            for y in range(b_ysize):
                for x in range(b_xsize):
                    if shape[x][y][z] in 'bs':
                        cnt_to_place += 1
            for y in range(b_ysize):
                line = ''
                for x in range(b_xsize):
                    line += shape[x][y][z]
                print(line)
            print('----------------------------------------------------------------')
            input()
    data.append(-1)
    return data



printers = [
    {'name': 'ItchyTec Cartesian 3D Printer', 'build_volume': np.array(
        [16, 16, 16]), 'type': '3axis', 'extruders': ['block', 'support'], 'slicers': [legacy]},
    {'name': 'ItchyTec ATGP 3D Printer', 'build_volume': np.array(
        [16, 16, 16]), 'type': '3axis', 'extruders': ['block', 'support'], 'slicers': [legacy]},
    # {'name': 'Gigachad 3D Printer', 'build_volume': np.array(
    #     [64, 256, 64]), 'type': '3axis', 'extruders': ['block', 'support'], 'slicers': [cartesian_standard]},
    {'name': 'ItchyTec 5-axis Drone Printer', 'build_volume': np.array([float('inf'), float('inf'), float('inf')]), 'type': '5axis', 'extruders': ['block', 'support'], 'slicers': [drone_5axis]},
    {'name': 'MTech Tower', 'build_volume': np.array([256, 256, 256]), 'type': '3axis', 'extruders': ['support', 'block'], 'slicers': [tower]},
    {'name': 'C.O.G. Inc. Warp Printer', 'build_volume': np.array([64, 64, 64]), 'type': '3axis', 'extruders': ['generic'], 'slicers': [warp_slicer]},
]


selected_printer = 1


class Manager(object):
    def new_thread(self, appobj):
        return Worker(parent=self, appobj=appobj)

    def on_thread_finished(self, thread, data, appobj):
        global voxels
        voxels = data
        print('thread finished')
        # print(voxels)
        vox_mesh = voxels.as_boxes()
        appobj.glapp.update_triangles(vox_mesh)
        appobj.slice_button.configure(state='normal')
        # voxels.show()


class Worker(Thread):

    def __init__(self, parent=None, appobj=None):
        self.parent = parent
        self.appobj = appobj
        super(Worker, self).__init__()

    def run(self):
        global scale, working_mesh
        print('begin voxelization')
        max_width = scale
        bounds = working_mesh.bounds
        size = bounds[1] - bounds[0]
        s_x = size[0]/(min(max_width, printers[selected_printer]['build_volume'][0])-1)
        s_y = size[1]/(min(max_width, printers[selected_printer]['build_volume'][1])-2)
        s_z = size[2]/(min(max_width, printers[selected_printer]['build_volume'][2])-1)
        sc = max(s_x, s_y, s_z)
        voxels = voxelize(working_mesh, sc)
        cur_scale = self.appobj.vox_scale_input.get()
        v_scale = max_width
        size = voxels.shape
        print(size, v_scale)
        v_scale = min(v_scale, max(size[0]+1, size[1]+1, size[2]+1))
        print(cur_scale, v_scale)
        if cur_scale != v_scale:
            scale = v_scale
            self.appobj.vox_scale_input.delete(0, 'end')
            self.appobj.vox_scale_input.insert(0, str(int(v_scale)))
        print('end voxelization')
        self.parent and self.parent.on_thread_finished(
            self, voxels, self.appobj)

cam_zoom = 5


class AppOgl(OpenGLFrame):
    def __init__(self, master):
        self.master = master
        super(AppOgl, self).__init__(master)
    def initgl(self):
        global VERTEX_SHADER, FRAGMENT_SHADER, mesh
        """Initalize gl states when the frame is created"""
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 1.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragmentshader = shaders.compileShader(
            FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.shaderProgram = shaders.compileProgram(
            vertexshader, fragmentshader)
        # self.bg_color = [0.0, 0.0, 0.0, 1.0]
        self.start = time.time()
        self.nframes = 0
        #   -0.5, -0.5, 0.0,
        #   -0.5, 0.5, 0.0,
        #   0.0, 1, 0.0]
        # self.triangles = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0])
        # m = trimesh.creation.box()
        self.master.reload()
        # mesh = trimesh.load(DEFAULT_MODEL)
        # mesh.vertices -= mesh.center_mass
        # self.update_triangles(mesh)
        print('e')
        print('current thread in initgl: ',
              threading.currentThread())
        self.position_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_buffer)
        # glBufferData(GL_ARRAY_BUFFER, self.triangles.nbytes,
        #              self.triangles, GL_STATIC_DRAW)
        position = glGetAttribLocation(self.shaderProgram, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position)
        self.projection = glGetUniformLocation(
            self.shaderProgram, 'projection')
        self.modelview = glGetUniformLocation(
            self.shaderProgram, 'modelview')
        self.normaltransform = glGetUniformLocation(
            self.shaderProgram, 'normaltransform')
        
        self.normal_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        # glBufferData(GL_ARRAY_BUFFER, self.triangles.nbytes,
        #              self.triangles, GL_STATIC_DRAW)
        normal = glGetAttribLocation(self.shaderProgram, 'normal')
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(normal)

        glDepthRange(0.0, 1.0)
        # glDisable(GL_CLIP_PLANE0)
        # glDisable(GL_CLIP_PLANE1)
        # glDisable(GL_CLIP_PLANE2)
        # glDisable(GL_CLIP_PLANE3)
        # glDisable(GL_CLIP_PLANE4)
        # glDisable(GL_CLIP_PLANE5)

    def update_triangles(self, mesh):
        m = mesh.copy()
        bounds = m.bounding_box.bounds
        size = bounds[1] - bounds[0]
        rescale = 4/max(size[0], size[1], size[2])
        m.apply_scale(rescale)
        triangles = []
        for face in m.faces:
            for vertex in face:
                triangles += (m.vertices[vertex]).tolist()
        # print(triangles)
        normals = m.face_normals
        norms = []
        for n in normals:
            norms.extend(n)
            norms.extend(n)
            norms.extend(n)
        self.normals = np.array(norms, dtype=np.float32)
        self.triangles = np.array(
            triangles, dtype=np.float32)


    def redraw(self):
        global cam_pitch, cam_yaw, cam_offset, target_offset, targeting_end, cam_zoom
        """Render a single frame"""
        if self.triangles is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.position_buffer)
            glBufferData(GL_ARRAY_BUFFER, self.triangles.nbytes,
                         self.triangles, GL_STATIC_DRAW)
            self.number_of_triangles = len(self.triangles) // 3
            self.triangles = None
            glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
            glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        if time.time() < targeting_end:
            cam_offset = [cam_offset[0]*(1-APPR_SPEED) + target_offset[0]*APPR_SPEED,
                          cam_offset[1]*(1-APPR_SPEED) +
                          target_offset[1]*APPR_SPEED,
                          cam_offset[2]*(1-APPR_SPEED) + target_offset[2]*APPR_SPEED]
            cam_yaw = cam_yaw*(1-APPR_SPEED) + target_yaw*APPR_SPEED
            cam_pitch = cam_pitch*(1-APPR_SPEED) + target_pitch*APPR_SPEED
            cam_zoom = cam_zoom*(1-APPR_SPEED) + target_zoom*APPR_SPEED
        glClearColor(*self.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shaderProgram)
        angle = time.time() / 3
        model_matrix = np.array((
            (math.cos(angle), 0, math.sin(angle), 0),
            (0, 1, 0, 0),
            (-math.sin(angle), 0, math.cos(angle), 0),
            (0, 0, 0, 1),
        ))
        projection_matrix = generate_projection_matrix(
            45, self.width / self.height, 1, 100)
        # cam_yaw = time.time()
        # cam_pitch = math.sin(time.time())
        view_matrix = generate_view_matrix(
            np.array([math.cos(cam_pitch)*math.cos(-cam_yaw+math.pi/2)*cam_zoom+cam_offset[0], math.sin(cam_pitch)*-cam_zoom+cam_offset[1], math.cos(cam_pitch)*math.sin(-cam_yaw+math.pi/2)*cam_zoom+cam_offset[2]]), cam_pitch, cam_yaw)
        normaltransform = np.copy(view_matrix)
        normaltransform[0][3] = 0
        normaltransform[1][3] = 0
        normaltransform[2][3] = 0

        # print(cam_yaw % (math.pi*2))
        # matrix = numpy.matmul(projection_matrix, np.matmul(
        #     view_matrix, rotation_matrix))
        # print(projection_matrix)
        # print(view_matrix)
        # print('---')
        # print(np.dot(matrix, np.array([0, -0.5, -0.5, 1])))
        # print(np.dot(matrix, np.array([0, -0.5, 0.5, 1])))
        # matrix = generate_view_matrix(
        #     np.array([0, 0, 0]), 0, math.sin(time.time()))
        # np.matmul(view_matrix, model_matrix))
        glUniformMatrix4fv(self.modelview, 1, GL_TRUE, view_matrix)
        glUniformMatrix4fv(self.projection, 1, GL_TRUE, projection_matrix)
        glUniformMatrix4fv(self.normaltransform, 1, GL_TRUE, normaltransform)
        glDrawArrays(GL_TRIANGLES, 0, self.number_of_triangles)
        glUseProgram(0)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # # draw triangle
        # GL.glBegin(GL.GL_TRIANGLES)
        # GL.glColor3f(1.0, 0.0, 0.0)
        # GL.glVertex2f(0.0, 0.0)
        # GL.glColor3f(0.0, 1.0, 0.0)
        # GL.glVertex2f(0.5, 0.5)
        # GL.glColor3f(0.0, 0.0, 1.0)
        # GL.glVertex2f(0.5, -0.5)
        # GL.glEnd()
        tm = time.time() - self.start
        self.nframes += 1
        # print("fps", self.nframes / tm, end="\r")


def generate_view_matrix(eye, pitch, yaw):
    """Generate a view matrix from the given eye position and angles"""
    # calculate the direction vector
    sinPitch = math.sin(pitch)
    cosPitch = math.cos(pitch)
    sinYaw = math.sin(yaw)
    cosYaw = math.cos(yaw)
    # xaxis = np.array([cosYaw, 0, -sinYaw])
    # yaxis = np.array([0, 1, 0])
    # zaxis = np.array([sinYaw, 0, cosYaw])
    xaxis = np.array([cosYaw, 0, -sinYaw])
    yaxis = np.array([sinYaw * sinPitch, cosPitch, cosYaw * sinPitch])
    zaxis = np.array([sinYaw * cosPitch, -sinPitch, cosPitch * cosYaw])
    # return [
    #     cosYaw, 0, sinYaw, 0,
    #     0, 1, 0, 0,
    #     -sinYaw, 0, cosYaw, 0,
    #     0, 0, 0, 1]
    # return np.matrix((
    #     (1, 0, 0, 0),
    #     (0, 1, 0, 0),
    #     (0, 0, 1, 0),
    #     (0, 0, 0, 1),
    # ))
    return np.array((
        (xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)),
        (yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)),
        (zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)),
        (0, 0, 0, 1)))
    # return np.array((
    #     (xaxis[0], yaxis[0], zaxis[0], 0),
    #     (xaxis[1], yaxis[1], zaxis[1], 0),
    #     (xaxis[2], yaxis[2], zaxis[2], 0),
    #     (-np.dot(xaxis, eye), -np.dot(yaxis, eye), -np.dot(zaxis, eye), 1)))


def generate_projection_matrix(fovy, aspect, near, far):
    if fovy <= 0 or fovy >= 180 or aspect <= 0 or near >= far or near <= 0:
        raise RuntimeError(
            "Invalid parameters to create perspective matrix")
    else:
        s = 1 / math.tan(math.radians(fovy)/2)
        return np.array((
            (s/(aspect**0.5), 0, 0, 0),
            (0, s*(aspect**0.5), 0, 0),
            (0, 0, (far + near) / (near - far), -1),
            (0, 0, far * near / (near - far), 0),
        ))


pos_birth = 0
prev_pos = 0
cam_offset = [0, 0, 0]
target_offset = [0, 0, 0]
target_zoom = 10
target_yaw = math.pi/4
target_pitch = -math.pi/6
targeting_end = time.time()+15


def go_home(event):
    global targeting_end
    targeting_end = time.time()+10


def rotate_camera(event):
    global pos_birth, prev_pos, cam_yaw, cam_pitch, targeting_end
    targeting_end = 0
    if time.time()-pos_birth < 0.01:
        dx = event.x-prev_pos[0]
        dy = event.y-prev_pos[1]
        cam_yaw += (event.x-prev_pos[0])*-0.01
        cam_pitch += (event.y-prev_pos[1])*-0.01
        cam_pitch = max(-math.pi/2+0.01, min(math.pi/2-0.01, cam_pitch))
        cam_yaw = cam_yaw % (math.pi*2)
    pos_birth = time.time()
    prev_pos = (event.x, event.y)


def move_camera(event):
    global pos_birth, prev_pos, cam_offset, targeting_end, cam_zoom
    targeting_end = 0
    if time.time()-pos_birth < 0.01:
        dx = (event.x-prev_pos[0])*cam_zoom/5
        dy = (event.y-prev_pos[1])*cam_zoom/5
        forward_vec = np.array(
            [-math.cos(-cam_yaw+math.pi/2), 0, -math.sin(-cam_yaw+math.pi/2)])
        right_vector = np.array(
            [-math.cos(-cam_yaw), 0, -math.sin(-cam_yaw)])*dx*0.005
        up_vector = (np.array([0, math.sin(cam_pitch+math.pi/2), 0]) +
                     forward_vec*math.cos(cam_pitch+math.pi/2))*dy*0.005
        cam_offset = [cam_offset[0]+right_vector[0]+up_vector[0],
                      cam_offset[1]+right_vector[1]+up_vector[1],
                      cam_offset[2]+right_vector[2]+up_vector[2]]
    pos_birth = time.time()
    prev_pos = (event.x, event.y)


def zoom_camera(event):
    global cam_zoom, targeting_end
    targeting_end = 0
    cam_zoom += event.delta*-0.001*cam_zoom
    cam_zoom = max(3, min(500, cam_zoom))


def validate(new_value):
    if new_value == "":
        return True
    if ' ' in new_value:
        return False
    try:
        int(new_value)
        return True
    except ValueError:
        return False

def names_of_slicers(slicers):
    for slicer in slicers:
        yield slicer(givename=True)

pzup = True
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Veloqity")
        self.geometry(f"{1280}x{720}")
        self.iconbitmap('icon.ico')
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.minsize(600, 300)

        self.sidebar_frame = customtkinter.CTkFrame(
            self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=8, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(32, weight=1)
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text='Veloqity', font=ctk.CTkFont(size=20, weight='bold'))
        self.logo_label.grid(row=0, column=0, padx=10, pady=(20, 10))
        self.load_button = ctk.CTkButton(
            self.sidebar_frame, text='Load File', command=self.select_3d_file)
        self.load_button.grid(row=1, column=0, padx=10, pady=10)
        self.printer_label = ctk.CTkLabel(
            self.sidebar_frame, text='Printer', font=ctk.CTkFont(size=15, weight='bold'))
        self.printer_label.grid(row=2, column=0, padx=10, pady=0)
        self.printer_info = ctk.CTkFrame(self.sidebar_frame)
        self.printer_info.grid(row=3, column=0, padx=10, pady=0)
        self.printer_info.grid_rowconfigure(8, weight=1)
        self.printer_name = ctk.CTkLabel(self.printer_info, text=printers[selected_printer]['name'])
        self.printer_name.grid(row=0, column=0, padx=10, pady=0)
        self.printer_build_volume = ctk.CTkLabel(self.printer_info, text=f"{printers[selected_printer]['build_volume'][0]}x{printers[selected_printer]['build_volume'][1]}x{printers[selected_printer]['build_volume'][2]}")
        self.printer_build_volume.grid(row=1, column=0, padx=10, pady=0)
        self.printer_type = ctk.CTkLabel(self.printer_info, text=printers[selected_printer]['type'])
        self.printer_type.grid(row=2, column=0, padx=10, pady=0)
        self.printer_switch = ctk.CTkButton(self.printer_info, text='Switch Printer', command=self.select_printer)
        self.printer_switch.grid(row=3, column=0, padx=10, pady=(0, 10))
        self.vox_scale_input = ctk.CTkEntry(
            self.sidebar_frame, validate='key', validatecommand=(self.register(validate), "%P"), placeholder_text="0.0")
        self.vox_scale_input.grid(row=4, column=0, padx=10, pady=10)
        for i in range(0, 10):
            self.bind(str(i), self.on_scale_changed)
        self.bind("<Delete>", self.on_scale_changed)
        self.bind("<BackSpace>", self.on_scale_changed)
        self.vox_view_toggle = ctk.CTkSwitch(self.sidebar_frame, text="Voxel View", command=self.reload)
        self.vox_view_toggle.grid(row=5, column=0, padx=10, pady=10)
        self.coordinate_flip = ctk.CTkSwitch(self.sidebar_frame, text="Y up?", command=self.reload)
        self.coordinate_flip.grid(row=6, column=0, padx=10, pady=10)
        self.slicers_list = list(names_of_slicers(printers[selected_printer]['slicers']))
        self.slicer_selected = ctk.StringVar(self)
        self.slicer_selected.set(self.slicers_list[0])
        self.slicer_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, values=self.slicers_list, variable=self.slicer_selected)
        self.slicer_dropdown.grid(row=7, column=0, padx=10, pady=10)
        self.slice_button = ctk.CTkButton(self.sidebar_frame, text='Slice', command=self.slice, state='disabled')
        self.slice_button.grid(row=8, column=0, padx=10, pady=10)

        bgc = [x/65635 for x in list(self.winfo_rgb(self.cget('bg')))]+[1.0]
        print(bgc)
        self.glapp = AppOgl(self)  # , width=1080, height=720)
        self.glapp.bg_color = bgc
        self.glapp.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.glapp.animate = 10
        self.glapp.bind('<B2-Motion>', move_camera)
        self.glapp.bind('<B3-Motion>', rotate_camera)
        self.glapp.bind('<MouseWheel>', zoom_camera)
        self.bind('<space>', go_home)
        # self.glapp.after(100, self.glapp.printContext)
        self.after(1, lambda: self.focus_force())

    def select_printer(self):
        popup = ctk.CTkToplevel()
        popup.iconbitmap('icon.ico')
        popup.transient(self)
        popup_w = 600
        popup_h = 800
        popup.geometry(
            f'{popup_w}x{popup_h}+{self.winfo_width()//2+self.winfo_x()-popup_w//2}+{self.winfo_height()//2+self.winfo_y()-popup_h//2}')
        popup.minsize(popup_w, popup_h)
        popup.title('Select Printer')
        popup.resizable(True, True)
        popup.grid_rowconfigure(8, weight=1)
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_columnconfigure(1, weight=1)
        for i, printer in enumerate(printers):
            printer_frame = ctk.CTkFrame(popup)
            printer_frame.grid(row=i//2, column=i % 2, padx=((10, 5) if i%2==0 else (5, 10)), pady=((10, 5) if i//2==0 else 5), sticky="ew")
            printer_frame.grid_rowconfigure(2, weight=1)
            printer_frame.grid_columnconfigure(0, weight=1)
            printer_name = ctk.CTkLabel(printer_frame, text=printer['name'])
            printer_name.grid(row=0, column=0, padx=10, pady=0)
            printer_build_volume = ctk.CTkLabel(printer_frame, text=f"{printer['build_volume'][0]}x{printer['build_volume'][1]}x{printer['build_volume'][2]}")
            printer_build_volume.grid(row=1, column=0, padx=10, pady=0)
            printer_type = ctk.CTkLabel(printer_frame, text=printer['type'])
            printer_type.grid(row=2, column=0, padx=10, pady=0)
            printer_select = ctk.CTkButton(printer_frame, text='Select', command=lambda printer_index=i: self.select_printer_callback(printer_index, popup))
            printer_select.grid(row=3, column=0, padx=10, pady=(0, 10))

    def select_printer_callback(self, printer_index, popup):
        global selected_printer, voxels
        selected_printer = printer_index
        self.slicers_list = list(names_of_slicers(printers[selected_printer]['slicers']))
        print(self.slicers_list)
        self.slicer_dropdown.configure(values=self.slicers_list)
        self.slicer_dropdown.set(self.slicers_list[0])
        popup.destroy()
        self.printer_name.configure(text=printers[selected_printer]['name'])
        self.printer_build_volume.configure(text=f"{printers[selected_printer]['build_volume'][0]}x{printers[selected_printer]['build_volume'][1]}x{printers[selected_printer]['build_volume'][2]}")
        self.printer_type.configure(text=printers[selected_printer]['type'])
        voxels = None
        self.reload()
    
    def slice(self):
        slicer = self.slicer_selected.get()
        slicer_index = None
        for i, slicer in enumerate(printers[selected_printer]['slicers']):
            if slicer(givename=True) == self.slicer_selected.get():
                slicer_index = i
                break
        slicer = printers[selected_printer]['slicers'][slicer_index]
        export_data = slicer(voxels=voxels, printer=printers[selected_printer])
        path = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Scrap Mechanic\\Data\\Importer"
        if not os.path.exists(path):
            os.makedirs(path)
        print(export_data)
        with open(path + "\\Importer.json", "w") as f:
            f.write(json.dumps(export_data))
        # with open('export_data.json', 'w') as f:
        #     f.write(json.dumps(export_data[0]))
        return
    
    def show_popup(self, title, body):
        popup = ctk.CTkToplevel()
        popup.iconbitmap('icon.ico')
        popup.transient(self)
        popup_w = 300
        popup_h = 100
        popup.geometry(
            f'{300}x{100}+{self.winfo_width()//2+self.winfo_x()-popup_w//2}+{self.winfo_height()//2+self.winfo_y()-popup_h//2}')
        popup.title(title)
        popup.resizable(False, False)
        label = ctk.CTkLabel(popup, text=body, justify='center')
        label.pack(padx=10, pady=10)
        button = ctk.CTkButton(
            popup, text='OK', command=popup.destroy)
        button.pack(padx=10, pady=10)
        popup.wait_visibility()
        popup.grab_set()
        popup.wait_window()

    def select_3d_file(self):
        global mesh, scale, voxels
        filename = ctk.filedialog.askopenfilename(
            title='Select 3D File', filetypes=(('3D object', '*.obj;*.stl;*.ply;*.glb'), ('All files', '*.*')))
        if filename:
            extension = filename.split('.')[-1]
            if extension in ['obj', 'stl', 'ply', 'glb']:
                try:
                    mesh = trimesh.load(filename)
                    mesh.vertices -= mesh.center_mass
                    voxels = None
                    printer = printers[selected_printer]
                    scale = max(printer['build_volume'][0], printer['build_volume'][1], printer['build_volume'][2])
                    self.vox_scale_input.delete(0, 'end')
                    self.vox_scale_input.insert(0, str(int(scale)))
                    self.reload()
                except Exception as e:
                    self.show_popup(
                        title='Error', body=f'Failed to load file:\n{e}')
                    # self.load_button.configure(text='Error; Load File')
            else:
                self.show_popup(
                    title='Error', body='Unsupported file format:\n.{extension})')

    def on_scale_changed(self, event):
        print(event)
        # Print a message to the console
        global scale, voxels
        pscale = scale
        need_up = False
        try:
            scale = float(self.vox_scale_input.get())
            need_up = True
        except:
            pass
        if scale <= 0:
            scale = pscale
            need_up = False
        if need_up:
            voxels = None
            self.reload()

    def reload(self):
        global mesh, voxels, scale, working_mesh, pzup
        vox_view = self.vox_view_toggle.get()
        z_up = not self.coordinate_flip.get()
        if z_up != pzup:
            voxels = None
        pzup = z_up
        if self.vox_scale_input.get() == '':
            scale = 16
            self.vox_scale_input.delete(0, 'end')
            self.vox_scale_input.insert(0, str(int(scale)))

        if mesh is None:
            mesh = trimesh.load(DEFAULT_MODEL)
            mesh.vertices -= mesh.center_mass
        working_mesh = mesh.copy()
        if z_up:
            working_mesh.vertices[:, [0, 1, 2]] = working_mesh.vertices[:, [1, 2, 0]]
        if vox_view:
            print('voxels', voxels)
            if voxels == None:
                print('voxelizing')
                mgr = Manager()
                thread = mgr.new_thread(self)
                thread.start()
            else:
                print('using existing voxelization')
                vox_mesh = voxels.as_boxes()
                self.glapp.update_triangles(vox_mesh)
        else:
            self.glapp.update_triangles(working_mesh)
        if voxels == None:
            self.slice_button.configure(state='disabled')

if __name__ == "__main__":
    ctk.set_appearance_mode('System')
    app = App()
    app.mainloop()
