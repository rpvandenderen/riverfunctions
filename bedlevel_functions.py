import numpy as np
import pandas as pd
import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import fiona
import multiprocessing
from functools import partial
import itertools



def chunks(data, n):
    """Yield successive n-sized chunks from a slice-able iterable."""
    for i in range(0, len(data), n):
        yield data[i:i+n]

def zonal_stats_partial(feats,tif,stats):
    """Wrapper for zonal stats, takes a list of features"""
    return zonal_stats(feats, tif, all_touched=False,stats=stats)

def bedraster2bedgrid(grid,rasterpath,rasternamelist,outputdir,datedir=None,covrat=.9,cores=int(multiprocessing.cpu_count()*.75)):
    gridname=os.path.basename(grid).split('.')[0]
    if not os.path.isdir(os.path.join(outputdir,gridname)):
        os.mkdir(os.path.join(outputdir,gridname))
    
    with fiona.open(grid) as src:
        features = list(src)
    p = multiprocessing.Pool(cores)
    with rasterio.open(os.path.join(rasterpath,rasternamelist[0] + '.tif')) as rast:
        nullval = rast.nodata
                
    for row in rasternamelist:
        #stats_lists=zonal_stats_partial(features,os.path.join(datedir,row + ".tif"),stats=['mean','count','nodata'])
        stats_lists= p.map(partial(zonal_stats_partial,tif=os.path.join(rasterpath,row + ".tif"),stats=['mean','count','nodata']),chunks(features, cores))
        stats = list(itertools.chain(*stats_lists))
        for ii in range(0,len(stats)):
            if (stats[ii]['count']/(stats[ii]['count']+stats[ii]['nodata'])<covrat)|(stats[ii]['mean']==None):
                stats[ii]['mean']=nullval
        means = [int(stat['mean']) for stat in stats]
        print(row + 'date')

        if datedir is not None:
            stats_lists= p.map(partial(zonal_stats_partial,tif=os.path.join(datedir,row + ".tif"),stats=['majority']),chunks(features, cores))
            stats = list(itertools.chain(*stats_lists))
    
            datenum =np.array( [stat['majority'] for stat in stats])
            datenum[(datenum==None)|(means==nullval)]=nullval
            
        
        with fiona.open(grid) as src:
            schema = src.schema
            # add the mean field to the schema of the resulting shapefile
            schema['properties']['etamean'] = 'int:32'
            if datedir is not None:
                schema['properties']['datenum'] = 'int:32'
            with fiona.open(os.path.join(outputdir,gridname,row + '.shp'), 'w', 'ESRI Shapefile', schema) as output:
                for i, feature in enumerate(src):
                    feature['properties']['etamean']= means[i]
                    if datedir is not None:
                        feature['properties']['datenum'] = datenum[i]
                    output.write(feature)
            print(row)
    return True 

def bedgrid2hdf(rasternamelist,outputHDF,nullval=np.iinfo(np.int32).min):
    if not os.path.isdir(outputHDF.replace(outputHDF.split(os.path.sep)[-1],'')):
        os.mkdir(outputHDF.replace(outputHDF.split(os.path.sep)[-1],''))
    
    
    df_all=pd.DataFrame([],columns=["redname",'datenum',"rkm",'m','n','etamean'])
    for rasname in rasternamelist:
        df_single=gpd.read_file(rasname)
        df_single=df_single.drop(df_single.columns.difference(['datenum','rkm','m','n','etamean']), 1).replace(nullval,np.nan)
        df_single['redname']=rasname.split(os.path.sep)[-1].replace('.shp','')
        df_all=pd.concat([df_all,df_single])
        print(rasname)
    
    df_all=df_all.merge(df_all.groupby('redname').min()['datenum'].rename('startdat'),on='redname')
    df_all=df_all.merge(df_all.groupby('redname').max()['datenum'].rename('enddat'),on='redname')
    df_all=df_all.astype({'rkm':'int64','m': 'int64','n': 'int64'})
    df_all.to_hdf(outputHDF,key='df_all',mode='w',complevel=9,complib='zlib')    
    return df_all