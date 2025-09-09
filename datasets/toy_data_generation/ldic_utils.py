import numpy as np
import json, jsonlines
from PIL import Image
import os, shutil
import pylidc as pl
from pylidc.utils import consensus
import tqdm
import jlc
from pathlib import Path
import pydicom as dicom
import warnings
from skimage.measure import find_contours
import matplotlib.pyplot as plt

def load_slices(scan,slice_idx,return_filenames=False):
    p = Path(scan.get_path_to_dicom_files())
    filenames = sorted(list(p.glob("*.dcm")))
    if isinstance(slice_idx,slice):
        slice_idx = range(*slice_idx.indices(len(filenames)))
    n = len(filenames)
    if return_filenames:
        slices = [filenames[n-1-i] for i in slice_idx]
    else:
        slices = [dicom.dcmread(filenames[n-1-i]).pixel_array for i in slice_idx]
    return slices

def slice_to_bbox(cbbox):
    return np.array([cbbox[0].start,cbbox[0].stop,cbbox[1].start,cbbox[1].stop])
def bbox_to_slice(bbox):
    return (slice(bbox[0],bbox[1],None),slice(bbox[2],bbox[3],None))

""" GLOBAL STRUCTURE
deepest_level: image_dict = {masks: [mask_id_0, mask_id_1, ...], image: x, mask_bbox: x, slice: x}
[   
    {
    scan_id0: bla,
    patient_id: bla, 
    metadata1: bla,
    metadata2: bla,
    nodules:[#nodules
                [#slices
                    image_dict0,
                    image_dict1,
                    ...
                ],
                [#slices
                ...
                ]
            ]
    }
]

"""

def load_data_from_scan(scan,
                        include_metadata = ["slice_thickness",
                                            "pixel_spacing",
                                            "patient_id",
                                            "id"
                                            ],
                        raw_images=False,
                        crop_bbox_per_slice=False):
    images = []
    mask_list = []
    
    scan_data = {**{k: v for k,v in scan.__dict__.items() if k in include_metadata},
                    "nodules": []}
    nods = scan.cluster_annotations()
    for nod_idx,anns in enumerate(nods):
        _,cbbox,masks = consensus(anns, clevel=0.5)
        if len(masks)==0:
            warnings.warn(f"no masks found for nodule {nod_idx}")
            continue
        assert all([masks[0].shape==mask.shape for mask in masks])
        
        slices = load_slices(scan, cbbox[2], return_filenames=raw_images)
        scan_data["nodules"].append([])
        for i in range(len(slices)):
            masks_i = [mask[:,:,i] for mask in masks]
            if not any([mask.any() for mask in masks_i]):
                continue
            m = len(mask_list)
            if crop_bbox_per_slice:
                bbox = get_slice_bbox(np.concatenate([m[...,None] for m in masks_i],axis=2))
                masks_i = [m[bbox[0]:bbox[1],bbox[2]:bbox[3]] for m in masks_i]
                bbox = [
                    bbox[0]+cbbox[0].start,
                    bbox[1]+cbbox[0].start,
                    bbox[2]+cbbox[1].start,
                    bbox[3]+cbbox[1].start
                ]
                bbox = [int(val) for val in bbox]
            else:
                bbox = slice_to_bbox(cbbox).tolist()
            image_dict = {"bbox": bbox,
                          "slice": int(i+cbbox[2].start),
                          "image_id": len(images),
                          "masks_id": list(range(m,m+len(masks)))}
            images.append(slices[i])
            scan_data["nodules"][-1].append(image_dict)
            mask_list.extend(masks_i)
    return images, mask_list, scan_data

def save_dict_list_to_json(data_list, file_path, append=False):
    assert isinstance(file_path,str), "file_path must be a string"
    if not isinstance(data_list,list):
        data_list = [data_list]
    
    if file_path.endswith(".json"):
        loaded_data = []
        if append:
            if Path(file_path).exists():
                loaded_data = load_json_to_dict_list(file_path)
                if not isinstance(loaded_data,list):
                    loaded_data = [loaded_data]
        data_list = loaded_data + data_list
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, indent=4)
    else:
        assert file_path.endswith(".jsonl"), "File path must end with .json or .jsonl"
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)

def load_json_to_dict_list(file_path):
    assert len(file_path)>=5, "File path must end with .json"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if file_path[-5:] == ".json":
        with open(file_path, 'r') as json_file:
            data_list = json.load(json_file)
    elif file_path[-6:] == ".jsonl":
        data_list = []
        with jsonlines.open(file_path) as reader:
            for line in reader:
                data_list.append(line)
    return data_list

def process_all_scans(scans,filename="data.jsonl",raw_images=False,crop_bbox_per_slice=False):
    imgname = lambda id: f"./images/img_{id:06d}.png"
    maskname = lambda id: f"./masks/mask_{id:06d}.png"
    if not Path("./masks").exists():
        Path("./masks").mkdir()
    if not Path("./images").exists():
        Path("./images").mkdir()
    global_image_id = 0
    global_mask_id = 0
    for scan in tqdm.tqdm(scans):
        images, mask_list, scan_data = load_data_from_scan(scan, raw_images=raw_images,
                                                           crop_bbox_per_slice=crop_bbox_per_slice)
        bboxes = []
        #add global ids
        for nodule in scan_data["nodules"]:
            for image_dict in nodule:
                image_dict["image_id"] += global_image_id
                image_dict["masks_id"] = [m + global_mask_id for m in image_dict["masks_id"]]
                bboxes.extend([image_dict["bbox"] for _ in image_dict["masks_id"]])
        save_dict_list_to_json(scan_data, filename, append=True)
        for image in images:
            imagename_i = imgname(global_image_id)
            if raw_images:
                moved_filename = imagename_i.replace(".png",".dcm")
                #copy file instead of saving
                shutil.copyfile(image,moved_filename)
            else:
                image = np.round(jlc.quantile_normalize(image,0.001)*255).astype(np.uint8)
                Image.fromarray(image).save(imagename_i)
            global_image_id += 1
        assert len(bboxes) == len(mask_list), f"len(bboxes)={len(bboxes)} != len(mask_list)={len(mask_list)}"
        for mask,bbox in zip(mask_list,bboxes):
            assert mask.shape == (bbox[1]-bbox[0],bbox[3]-bbox[2]), f"mask shape {mask.shape} does not match bbox {bbox} with shape {(bbox[1]-bbox[0],bbox[3]-bbox[2])}"
            mask = (mask*255).astype(np.uint8)
            Image.fromarray(mask).save(maskname(global_mask_id))
            global_mask_id += 1


def load_image_mask(scan_id, image_id, mask_id, data):
    if isinstance(mask_id,int):
        mask_id = [mask_id]
    assert scan_id < len(data), "scan_id out of range"
    found_image_dict = False
    image_ids = []
    for nod in data[scan_id]["nodules"]:
        for image_dict in nod:
            image_ids.append(image_dict["image_id"])
            if image_dict["image_id"]==image_id:
                found_image_dict = True
                break
        if found_image_dict:
            break
    assert found_image_dict, f"image_id not found in scan. image_ids found: {image_ids}"

    image_path = f"./images/img_{image_id:06d}.png"
    image = np.array(Image.open(image_path))
    h,w = image.shape
    mask = np.zeros((h,w,len(mask_id)),dtype=np.uint8)
    bbox = image_dict["bbox"]
    for i,mask_id in enumerate(mask_id):
        mask_path = f"./masks/mask_{mask_id:06d}.png"
        s1,s2 = slice(*bbox[:2]),slice(*bbox[2:])
        mask[s1,s2,i] = np.array(Image.open(mask_path))
    return image,mask,bbox

def get_slice_bbox(mask):
    assert np.any(mask), "mask is empty"
    if len(mask.shape)==3:
        mask = mask.any(axis=-1)
    b1,b2 = np.where(mask.any(axis=1))[0][[0,-1]]
    b3,b4 = np.where(mask.any(axis=0))[0][[0,-1]]
    return b1,b2+1,b3,b4+1

def bbox_crop(image,mask,bbox,size,mask_pad=0,image_pad=0):
    """crops centrally around the bbox, adding enough pixels to reach
    the desired size. If the larger size exceeds the image size, the
    image is padded with zeros. if the number of needed pixels is odd,
    one more pixel is added to the top left corner (smaller index)"""
    if isinstance(size,int):
        d1,d2 = size,size
    else:
        d1,d2 = size
    if bbox is None:
        bbox = get_slice_bbox(mask)
    b1,b2,b3,b4 = bbox # image[b1:b2,b3:b4] is the bbox
    h,w = image.shape
    assert mask.shape[:2]==(h,w), f"mask shape {mask.shape} does not match image shape {image.shape}"
    #number of available pixels on each side
    avail = b1,h-b2,b3,w-b4
    assert all([val>=0 for val in avail]), f"bbox is invalid since it exceeds bounds: {bbox}. Image shape: {image.shape}"
    assert b1<b2 and b3<b4, f"bbox is invalid since we dont have (stop > start): {bbox}"
    #number of additional pixels needed on each side
    a = (d1-(b2-b1))/2, (d2-(b4-b3))/2
    c = lambda x: int(np.ceil(x))
    f = lambda x: int(np.floor(x))
    a = [c(a[0]),f(a[0]),c(a[1]),f(a[1])]
    #number of additional pixels needed on each side, inside the image
    a_i = [min(a[i],avail[i]) for i in range(4)]
    #number of additional pixels needed on each side, which must be padded since they are outside the image
    a_o = [val1-val2 for val1,val2 in zip(a,a_i)]
    #old_slice
    old_slice = (slice(b1-a_i[0],b2+a_i[1]),slice(b3-a_i[2],b4+a_i[3]))
    #new_slice
    new_slice = (slice(a_o[0],d1-a_o[1]),slice(a_o[2],d2-a_o[3]))
    image_new = np.zeros((d1,d2),dtype=image.dtype)+image_pad
    image_new[new_slice] = image[old_slice]
    mask_shape_new = list(mask.shape)
    mask_shape_new[0] = d1
    mask_shape_new[1] = d2
    mask_new = np.zeros(mask_shape_new,dtype=mask.dtype)+mask_pad
    mask_new[new_slice] = mask[old_slice]
    return image_new,mask_new


def plot_mask_contours(masks,colors=lambda j: f"C{j}",with_consensus=True):
    for j in range(len(masks)):
        first = True
        for c in find_contours(masks[j], 0.5):
            if first:
                first = False
                plt.plot(c[:,1], c[:,0], colors(j), label="Annotation %d" % (j+1))
            else:
                plt.plot(c[:,1], c[:,0], colors(j))

    if with_consensus:
        mean_mask = sum([(m>0).astype(float) for m in masks])/len(masks)
        for c in find_contours(mean_mask,0.5):
            plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')
