# misc
import sys
import os

# load/save files
import json

# plot
import matplotlib.pyplot as plt
from PIL import Image

# datascience libs
import numpy as np
import math


path_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
dataset_dir = os.path.join(path_, "datasets")
model_dir = os.path.join(path_, "models")





import random

class BinPackerNode:
    def __init__(self, x=0, y=0, width=0,height=0, data=None, left=None,right=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.data = data
        self.left = left
        self.right = right

    def split(self, data, width, height):
        self.data = data
        self.left = BinPackerNode(self.x,self.y+height, self.width, self.height-height)
        self.right = BinPackerNode(self.x+width,self.y, self.width-width, height)
        return self

    @staticmethod
    def find(node, width, height):
        find = BinPackerNode.find
        if node.data:
            return find(node.right, width, height) or find(node.left, width, height)
        elif width <= node.width and height <= node.height:
            return node
        return None

class BinPacker:
    def __init__(self, width, height, verbose=0, percentage_completion=0):
        self.root = BinPackerNode(0,0,width,height)
        self.width = width
        self.height = height
        self.verbose = verbose
        self.percentage_completion = percentage_completion

    cbsort = {
        "w": (lambda a,b: b["width"] - a["width"]),
        "h": (lambda a,b: b["height"] - a["height"]),
        "a": (lambda a,b: b["width"]*b["height"] - a["width"]*a["height"]),
        "max": (lambda a,b: max(b["width"], b["height"]) - max(a["width"], a["height"])),
        "min": (lambda a,b: min(b["width"], b["height"]) - min(a["width"], a["height"])),
        "random": (lambda a,b: random.random() - 0.5),
        "height": (lambda a,b: BinPacker.msort(a, b, ['h','w'])),
        "width": (lambda a,b: BinPacker.msort(a, b, ['w','h'])),
        "area": (lambda a,b: BinPacker.msort(a, b, ['a','h','w'])),
        "maxside": (lambda a,b: BinPacker.msort(a, b, ['max','min','h','w'])),
    }

    @staticmethod
    def msort(a, b, criteria):
        diff = 0
        for n in range(len(criteria)):
            diff = BinPacker.cbsort[criteria[n]](a,b)
            if diff != 0:
                break
        return diff

    @staticmethod
    def swap(a,i,j):
        t = a[i]
        a[i] = a[j]
        a[j] = t

    @staticmethod
    def sort(arr, criteria = ['height']):
        for i in range(0, len(arr)-1):
            for j in range(i+1, len(arr)):
                if BinPacker.msort(arr[i], arr[j], criteria) > 0:
                    BinPacker.swap(arr,i,j)


    def fit(self, blocks_src, criteria = ['height']):
        res = []
        blocks = []

        for i in range(len(blocks_src)):
            blocks.append(blocks_src[i])

        # if criteria doesn't exist, we assume that all boxes have the same size
        if not criteria == None:
            if self.verbose:
                print("Sorting nodes")
            BinPacker.sort(blocks, criteria)

        blocks_count = len(blocks)
        for i in range(blocks_count):
            block = blocks[i]
            w = block["width"]
            h = block["height"]
            node = BinPackerNode.find(self.root, w,h)
            if not node:
                # if criteria doesn't exist, we assume that all boxes have the same size
                if criteria == None:
                     break
                continue
            if not node.split(block["data"] if "data" in block else "empty", w,h):
                continue
            node.width = w
            node.height = h
            res.append(node)

            new_percentage = int((i * 100) / blocks_count)
            if new_percentage > self.percentage_completion:
                self.percentage_completion = new_percentage
                if self.verbose:
                    print("Insert nodes {}%".format(self.percentage_completion))
        return res


def emnist_get_mapping(filepath):
    min_index, max_index = 999999, 0
    data = {}
    with open(filepath) as f:
        while True:
            line = f.readline()
            if not line:
                break
            tmp = [int(s) for s in line.strip().split(' ') if s.isdigit()]
            data[tmp[0]] = tmp[1]
            min_index = min(min_index, tmp[0])
            max_index = max(max_index, tmp[0])
    # create array with size
    emnist_mapping = [-1] * ((max_index-min_index) + 1)
    # dict to array
    for key, val in data.items():
        emnist_mapping[key-min_index] = val
    return emnist_mapping

import random


def split_to_chunks(length, chunk_size=3):
    chunks = math.ceil(length/float(chunk_size))
    res = []
    if (chunk_size >= length):
        return [length]
    for i in range(chunks-1):
        res.append(chunk_size*(i+1) - (chunk_size*i))
    res.append(length - chunk_size*(i+1))
    return res

def area_to_rectangle(area, tile_width, tile_height):
    height = math.ceil(math.sqrt(area))
    while height > 1:
        if area % height == 0:
            break
        height -= 1
    return [int(area / height)*tile_width, int(height)*tile_height]

def area_to_grid_rectangle(area, tile_width, tile_height, max_tiles_width):
    #rest = area % width
    #arr = area_to_rectangle(area)
    #return [arr[0], arr[1]]
    print("area:", area)
    width = max_tiles_width * tile_width
    height = math.ceil(area / width)
    if height == 1:
        width = tile_width * area
    return [width, height*tile_height]


def size_to_pack(tiles_count, tile_width=28, tile_height=28,
                 max_width=256, max_height=256):

    while True:
      max_tiles_width = max_width // tile_width
      max_tiles_height = max_height // tile_height

      count = []
      splited = split_to_chunks(tiles_count, max_tiles_width * max_tiles_height)
      for i in range(len(splited)):
          count.append(area_to_rectangle(splited[i], tile_width, tile_height))

      # width or height must be < max_width
      # ratio of the last image must be < 2
      if np.amax(count) <= max_width and (count[-1][0] / count[-1][1]) < 2:
        break
      max_width -= tile_width
      max_height -= tile_height

    # last one
    #count.append(area_to_grid_rectangle(splited[-1], tile_width, tile_height, max_tiles_width))
    return {"count": count, "grid": splited}


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def draw_object(src, sx,sy, s_width,s_height,
                dst, dx,dy, d_width,d_height):
    dx -= sx
    dy -= sy
    for j in range(sy,s_height):
        for i in range(sx,s_width):
            color = src[j, i]
            dst[j+dy, i+dx] = color# if color else 127
    return

def list_first_val(arr):
    for i in arr:
        if i > 0:
            return 1
    return -1

def get_bbox(data):
    """aligned-axis bounding-box (bounding square)"""
    x1 = 0xffff
    y1 = 0xffff
    x2 = 0
    y2 = 0
    # y1
    for j in range(len(data)):
        if list_first_val(data[j]) > 0:
            y1 = j
            break
    # y2
    for j in range(len(data)):
        end = len(data)-j-1
        if list_first_val(data[end]) > 0:
            y2 = end
            break
    # x1, x2
    for j in range(len(data)):
        ydata = data[j]
        val = 0xffff
        last = 0
        for i in range(len(ydata)):
            if ydata[i] > 0:
                x1 = min(x1,i)
                x2 = max(x2,i)
    return [x1,y1, x2+1,y2+1]

def path_basename_with_dir(filepath):
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(dirname)
    return os.path.join(basename, os.path.basename(filepath))

def emnist_convert(dataset_dir, input_name, output_dir, emnist_mapping, X_set='train',
                   max_width=4096, max_height=4096):
    X_set_dir = os.path.join(dataset_dir, output_dir)
    makedirs(X_set_dir)
    input_file = os.path.join(dataset_dir, input_name+'-'+X_set+'.csv')

    tile_width = 28
    tile_height = 28
    blocks = []
    json_output = {
        "width": tile_width,
        "height": tile_height,
        "files": [],
        "mapping": emnist_mapping,
        "id": [],
        "bbox": [],
    }

    print("Read data")
    i = 0
    with open(input_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            tmp = [int(s) for s in line.strip().split(',') if s.isdigit()]
            pixels = np.array(tmp[1:], dtype='uint8').reshape(tile_height,tile_width)
            pixels = np.flip(pixels, axis=1)
            pixels = np.rot90(pixels, k=1, axes=(0, 1))
            json_output['id'].append(tmp[0])
            json_output['bbox'].append(get_bbox(pixels))
            blocks.append({
                "width": tile_width,
                "height": tile_height,
                "data": {
                    "idx": i,
                    "pixels": pixels
                }
            })
            i += 1
            #if i > 200:
            #    break

    print("Build tree")
    packs = size_to_pack(tiles_count=len(blocks),
                         tile_width=tile_width, tile_height=tile_height,
                         max_width=max_width, max_height=max_height)
    #packs['count'][2][1] += 28 * 2
    print(packs, len(blocks))
    count = 0
    pack_index = 0
    while len(blocks) > 0:
        percentage = packer.percentage_completion if 'packer' in locals() else 0
        image_width = packs['count'][pack_index][0]
        image_height = packs['count'][pack_index][1]
        pack_index += 1
        packer = BinPacker(image_width,image_height, verbose=1, percentage_completion = percentage)
        res = packer.fit(blocks, None)
        # advance on the next pack that couldn't be fitted
        blocks = blocks[len(res):]
        dst = np.zeros((packer.height,packer.width), dtype="uint8")
        count_prev = count
        for i in range(len(res)):
            node = res[i]
            if node.data == "empty":
                continue
            count += 1
            draw_object(node.data['pixels'],
                        0,0, tile_width,tile_height,
                        dst, node.x,node.y, packer.width,packer.height)
        # write image
        image_filename = "X_{}_{}_to_{}.webp".format(X_set,count_prev,count-1)
        image_filename = os.path.join(X_set_dir, image_filename)
        json_output['files'].append(os.path.basename(image_filename))
        print("Save:", path_basename_with_dir(image_filename))
        im = Image.fromarray(dst, mode='L')
        im.save(image_filename, format='webp', lossless = True)
        # plot image
        #plt.figure(figsize = (20,20))
        #plt.imshow(dst, cmap='gray')
        #plt.show()
    # write json file
    json_filename = os.path.join(X_set_dir, X_set + ".json")
    json_dump = json.dumps(json_output,separators=(',',':'))
    with open(json_filename, 'w', encoding='utf-8') as f:
        f.write(json_dump)
        print("Save:", path_basename_with_dir(json_filename))
    print("End")




if __name__ == "__main__":
    print("Old recursion limit:", sys.getrecursionlimit())
    sys.setrecursionlimit(3000)
    print("New recursion limit:", sys.getrecursionlimit())

    # test
    #size_to_pack(tiles_count=240000, tile_width=28, tile_height=28,
    #             max_width=4096, max_height=4096)

    # emnist-mnist (basic, only digits)
    set_name = 'emnist-mnist'
    emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    print("mapping:", emnist_mapping)
    emnist_convert(dataset_dir, set_name, ''+set_name, emnist_mapping, 'test')
    emnist_convert(dataset_dir, set_name, ''+set_name, emnist_mapping, 'train')

    # emnist-balanced
    #set_name = 'emnist-balanced'
    #emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    #print("mapping:", emnist_mapping)
    #emnist_convert(dataset_dir, set_name, ''+set_name, emnist_mapping, 'test')
    #emnist_convert(dataset_dir, set_name, ''+set_name, emnist_mapping, 'train')

    # emnist-digits (only digits)
    #set_name = 'emnist-digits'
    #emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    #print("mapping:", emnist_mapping)
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'test')
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'train')

    # emnist-letters (only letters)
    #set_name = 'emnist-letters'
    #emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    #print("mapping:", emnist_mapping)
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'test')
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'train')


    #set_name = 'emnist-byclass'
    #emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    #print("mapping:", emnist_mapping)
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'test')
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'train')


    #set_name = 'emnist-bymerge'
    #emnist_mapping = emnist_get_mapping(os.path.join(dataset_dir, set_name + '-mapping.txt'))
    #print("mapping:", emnist_mapping)
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'test')
    #emnist_convert(dataset_dir, set_name, set_name, emnist_mapping, 'train')
