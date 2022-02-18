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
#import descarteslabs as dl
import ee
from geemap import geojson_to_ee
from area import area
import pyproj
from scipy import ndimage
import numpy as np
from utils import *

logging.info(f'Initialising Earth Engine')
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
s2=ee.ImageCollection("COPERNICUS/S2_SR")
# data download from earth engine
#raster_client = dl.Raster()
#metadata_client = dl.Metadata()
trn_dltiles = json.load(open(os.path.join(os.getcwd(),'data','all_trn_dltiles.geojson'),'r'))['features']
trn_polygons = json.load(open(os.path.join(os.getcwd(),'data','all_trn_polygons.geojson'),'r'))['features']
#ee_trn=geojson_to_ee(trn_dltiles)
#ee_polygon=geojson_to_ee(trn_polygon)
print("importing training tiles and polygons completed")
def annotation_from_tile(tile_key,ii_t,mode='trn'):
    print(f'Fetching tile {tile_key}')
    tile = trn_dltiles[ii_t]
    tilesize=tile['properties']['tilesize']
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

    # get scenes for dltile
    scene=s2.filterBounds(ee_trn).filterDate(season_start[season],season_end[season]).filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', 20).sort('CLOUDY_PIXEL_PERCENTAGE').first()
    if not scenes:
        return None
    print("getting 10m bands as array")
    scene_part1=scene.select(['B2','B3','B4','B8'])
    tmp_arr1=geemap.ee_to_numpy(scene_part1,region=ee_trn)[0:tilesize,0:tilesize,:]
    print("getting 20m bands as array")
    scene_part2=scene.select(['B5','B6','B7','B8A','B11','B12')
    tmp_arr2=geemap.ee_to_numpy(scene_part2,region=ee_trn)[0:tilesize/2,0:tilesize/2,:]
    print("resampling 20m bands")
    tmp_arr2=ndimage.zoom(tmp_arr2,(2,2,1),order=0)
    print("merging both into single array")
    trn_tile=np.concatenate((tmp_arr1,tmp_arr2),axis=2)
    print(trainig array shape is"+trn_tile.shape)

    # get intersecting polygons
    tile_poly = geometry.Polygon(tile['geometry']['coordinates'][0])

    if mode=='trn':
        all_polygons = [geometry.shape(pp['geometry']) for pp in trn_polygons]
    else:
        all_polygons = [geometry.shape(pp['geometry']) for pp in test_polygons]

    intersect_polys = [pp for pp in all_polygons if pp.intersects(tile_poly)] 

    # clip tile arr [0.,1.]
    tile_arr = (tile_arr/255.).clip(0.,1.)


    # make an annotation array
    annotations = np.zeros((tilesize,tilesize)) #np.ones((arr.shape[0], arr.shape[1]))*128
    im = Image.fromarray(annotations, mode='L')
    draw = ImageDraw.Draw(im)
    # draw annotation polygons
    for pp in intersect_polys:
    print(pp)
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
    json.dump(fc_out,open('training/data/S2_unet/'+str(ii_t)+'.geojson','w'))
    np.savez('training/data/S2_unet/'+str(ii_t)+'.npz', data = tile_arr, annotation=annotations)


    return True

def multidownload(n_cpus,keys):
    pool = mp.Pool(n_cpus)
    pool.starmap(annotation_from_tile, list(zip(keys, range(len(keys)))))

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    keys = [tt['properties']['key'] for tt in trn_dltiles[0:2]]
    multidownload(2,keys)




