import json, re, glob, geojson, os, pickle, logging, sys
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from shapely import geometry
from shapely.affinity import affine_transform
from shapely.ops import transform
from PIL import Image, ImageDraw
import multiprocessing as mp
import os,site
site.addsitedir(os.path.abspath(os.path.join(__file__ ,"../../..")))
#import ee
import ee
from geemap import geojson_to_ee,ee_to_numpy
from scipy import ndimage
from area import area
import pyproj
from scipy import ndimage
import numpy as np
from utils import *
from random import shuffle
logging.info(f'Initialising Earth Engine')
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
s2=ee.ImageCollection("COPERNICUS/S2_SR")
# data download from earth engine
#raster_client = dl.Raster()
#metadata_client = dl.Metadata()
input_data="C:\\hpc\\data\\"
trn_dltiles = json.load(open(os.path.join(input_data,'all_trn_dltiles.geojson'),'r'))['features']
trn_polygons = json.load(open(os.path.join(input_data,'all_trn_polygons.geojson'),'r'))['features']
#ee_trn=geojson_to_ee(trn_dltiles)
#ee_polygon=geojson_to_ee(trn_polygon)
#print("importing training tiles and polygons completed")
output="C:\\hpc\\data\\training\\S2_unet\\"
tilesize=200
tilesize1=100
def maskclouds(image):
    band_qa = image.select('QA60')
    cloud_mask = ee.Number(2).pow(10).int()
    cirrus_mask = ee.Number(2).pow(11).int()
    mask = band_qa.bitwiseAnd(cloud_mask).eq(0) and(band_qa.bitwiseAnd(cirrus_mask).eq(0))
    return image.updateMask(mask).divide(10000)
     
def find_arr(scene,ee_trn):
    try:
        #print("iterator working")
        #print("selecting 10m bands")
        scene_part1=scene.select(['B2','B3','B4','B8'])
        #print("converting 10m bands as array")
        tmp_arr1=ee_to_numpy(scene_part1,region=ee_trn,default_value=0)
        if tmp_arr1 is None:
            return None
        fill_frac = np.sum(tmp_arr1[:,:,-1]>0)/tmp_arr1.shape[0]/tmp_arr1.shape[1]
        if fill_frac<0.8:
            return None
        tmp_arr1=tmp_arr1[0:tilesize,0:tilesize,:]
        #print("selecting 20m bands")
        scene_part2=scene.select(['B5','B6','B7','B8A','B11','B12'])
        #print("converting 20m bands as array")
        tmp_arr2=ee_to_numpy(scene_part2,region=ee_trn,default_value=0)
        if tmp_arr2 is None:
            return None
        tmp_arr2=tmp_arr2[0:tilesize1,0:tilesize1,:]
        #print("resampling 20m bands")
        tmp_arr2=ndimage.zoom(tmp_arr2,(2,2,1),order=0)
        #print("merging both into single array")
        trn_tile=np.concatenate((tmp_arr1,tmp_arr2),axis=2)
        return trn_tile
    except Exception as error:
        print(error)
        return None
def annotation_from_tile(tile_key,ii_t,mode='trn'):
    try:
        print(f'Fetching tile {ii_t}')
        tile = trn_dltiles[ii_t]
        ee_trn=geojson_to_ee({'type':'FeatureCollection','features':[tile]})
        # get a random season
        season = np.random.choice([0,1,2,3])

        season_start = {
            0:'2018-01-01',
            1:'2018-04-01',
            2:'2018-07-01',
            3:'2018-10-01'
        }
        season_end = {
            0:'2018-03-31',
            1:'2018-06-30',
            2:'2018-09-30',
            3:'2018-11-30'
        }
        # get intersecting polygons
        tile_poly = geometry.Polygon(tile['geometry']['coordinates'][0])

        if mode=='trn':
            all_polygons = [geometry.shape(pp['geometry']) for pp in trn_polygons]
        else:
            all_polygons = [geometry.shape(pp['geometry']) for pp in test_polygons]

        intersect_polys = [pp for pp in all_polygons if pp.intersects(tile_poly)] 
        #get scenes for dltile
        scene=s2.filterBounds(ee_trn).filterDate(season_start[season],season_end[season]).filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', 20).sort('CLOUDY_PIXEL_PERCENTAGE').map(maskclouds).first()
        if not scene.getInfo():
            #print("no scene found exiting")
            return None
        #print(scene.getInfo())
        trn_tile=find_arr(scene,ee_trn)
        if trn_tile is None:
            return None
        # make an annotation array
        annotations = np.zeros((tilesize,tilesize)) #np.ones((arr.shape[0], arr.shape[1]))*128
        im = Image.fromarray(annotations, mode='L')
        draw = ImageDraw.Draw(im)
        # draw annotation polygons
        for pp in intersect_polys:
            pp_intersection = pp.intersection(tile_poly)
            if pp_intersection.type == 'MultiPolygon':
                sub_geoms = list(pp_intersection)
            else:
                sub_geoms = [pp_intersection]

            for sub_geom in sub_geoms:
                xs, ys = sub_geom.exterior.xy
                draw.polygon(list(zip(xs, ys)), fill=255)
                for hole in sub_geom.interiors:
                    xs,ys = hole.xy
                    draw.polygon(list(zip(xs, ys)), fill=0)


        annotations = np.array(im)

        # output
        features = []
        features.append(geojson.Feature(geometry=tile_poly, properties=tile['properties']) )

        for p in intersect_polys:
            features.append(geojson.Feature(geometry=p, properties={}) )

        fc_out = geojson.FeatureCollection(features)

        logging.info(f'Writing data and annotation for {tile_key}')
        json.dump(fc_out,open(output+str(ii_t)+'.geojson','w'))
        np.savez(output+str(ii_t)+'.npz', data = trn_tile, annotation=annotations)
        return True
    except Exception as error:
        print(error)
        return None
def make_records(directory):
    """
    Makes a Pickle of a list of record dicts storing data and meta information
    """
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    meta_files = glob.glob(os.path.join(directory,'*.geojson'))

    records = []
    for npz in npz_files:
        ii = npz.split('/')[-1].split('.')[0]
        meta = [m for m in meta_files if m.split('/')[-1].split('.')[0]==ii][0]
        records.append({'data':npz, 'meta':meta})

    shuffle(records)
    pickle.dump(records, open(os.path.join(directory,'records.pickle'),'wb'))

def multidownload(n_cpus,keys):
    pool = mp.Pool(n_cpus)
    pool.starmap(annotation_from_tile, list(zip(keys, range(len(keys)))))

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    keys = [tt['properties']['key'] for tt in trn_dltiles]
    multidownload(24,keys)
    print("training data prep over, now making pickle record")
    make_records(output)
    print("Done")



