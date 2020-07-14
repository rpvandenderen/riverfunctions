import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import itertools
from operator import itemgetter
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


def check_gdf(gdf):
    if type(gdf) is str:
        gdf=gpd.read_file(gdf)
    elif (type(gdf) is list) and (type(gdf[0]) is str):
        gdfnames=gdf
        gdf=gpd.read_file(gdf[0])
        for ii in range(1,len(gdfnames)):
            gdf=pd.concat([gdf,gpd.read_file(gdfnames[ii])])
        gdf.reset_index(inplace=True)
            
    elif type(gdf) is not gpd.geodataframe.GeoDataFrame:
        ValueError('This type of input is not supported (yet)')
    return gdf

def compute_cent2point_rkm(geomcentline,kmpoints,strkm='KILOMETER',dx=10):
    # computes points with distance dx on center line of the river and assings river kilometers to the points
    shp_geom = check_gdf(geomcentline) 
    geometry=[shp_geom['geometry'][0].interpolate(ii) for ii in np.arange(0,int(shp_geom['geometry'].length[0])+dx,dx)]
    gdf_cent=gpd.GeoDataFrame([],geometry=geometry)
    
    km_pnt  = check_gdf(kmpoints) 
    gdf_cent['rkm']=0.0
    gdf_cent['dL_rkm']=0.0
    pts3 = gdf_cent.geometry.unary_union
    for ii in range(1,len(km_pnt)-1):
        idx=gdf_cent.index[(gdf_cent.geometry.geom_equals(nearest_points(km_pnt['geometry'].iloc[ii], pts3)[1]))][0]
        # idx=gdf_cent.distance(km_pnt['geometry'].iloc[ii]).idxmin()
        
        dL=gdf_cent.loc[idx,'geometry'].distance(km_pnt['geometry'].iloc[ii])
        
        if (gdf_cent['dL_rkm'].loc[idx]==0.0)|(gdf_cent['dL_rkm'].loc[idx]>dL):
            gdf_cent['dL_rkm'].at[idx]=dL
            dLsign=np.sign(gdf_cent.loc[idx,'geometry'].distance(km_pnt['geometry'].iloc[ii-1])-gdf_cent.loc[idx,'geometry'].distance(km_pnt['geometry'].iloc[ii+1]))
            gdf_cent['rkm'].at[idx]=km_pnt[strkm].iloc[ii]+dLsign*gdf_cent['dL_rkm'].loc[idx]/1000
    
    
    gdf_cent['dL']=np.cumsum(gdf_cent.distance(gdf_cent.shift(1)))
    idx=(gdf_cent['rkm']==0.0)
    gdf_cent['rkm'].loc[idx]=np.interp(gdf_cent['dL'].loc[idx],gdf_cent['dL'].loc[~idx],gdf_cent['rkm'].loc[~idx])
    gdf_cent=gdf_cent.dropna(axis=0).drop_duplicates(subset=['rkm'],keep=False).drop(labels=['dL', 'dL_rkm'],axis=1)
    return gdf_cent
   

def feat2point_rkm(feat,kmpoints,dx,geomcentline=None,strkm='KILOMETER'):
    
    if geomcentline is None:
        gdf_feat=compute_cent2point_rkm(feat,kmpoints,strkm=strkm,dx=dx)
        gdf_feat['dist']=0
        print('Centerline points created')
    else:
        gdf_cent=compute_cent2point_rkm(geomcentline,kmpoints,strkm=strkm,dx=min(10,dx/10))
        print('Centerline points created')
        
        shp_geom  = check_gdf(feat) 
        if shp_geom.geom_type.iloc[0] == 'LineString':
            geometry=[shp_geom['geometry'][0].interpolate(ii) for ii in np.arange(0,int(shp_geom['geometry'].length[0])+dx,dx)]
            gdf_feat=gpd.GeoDataFrame([],geometry=geometry)
        elif shp_geom.geom_type.iloc[0] == 'Point':
            gdf_feat=shp_geom
        else:
            ValueError('This type of feature is not supported (yet)')
        
        print('Computing riverkilometer')
        gdf_feat = near_rkm(gdf_feat, gdf_cent, 'rkm')

    return gdf_feat


def near_rkm(gdf_feat, gdf_cent, gdf_cent_cols=['rkm']):
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdf_feat.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdf_cent.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    if 'rkm' in gdf_cent_cols:
        gdf = pd.concat(
            [gdf_feat, gdf_cent.loc[idx, gdf_cent_cols].reset_index(drop=True),
             pd.Series(dist, name='dist')], axis=1)
    else:
        idx=list(idx)
        gdf = pd.concat([gdf_feat, gdf_cent[gdf_cent_cols].iloc[idx].set_index(gdf_feat.index)], axis=1)
    return gdf

def create_crossfeature(df_pnt,gridw,dy=None,feattype='Polygon'):
    if dy is None:
        dy=gridw
    if (feattype=='LineString')&(dy!=gridw):
        # this warning is added because it is unsure what kind of lines you would like
        dy=gridw
        Warning('dy!=gridw only outer bounds are used to create crosssections!')
        
          
    dL = np.linspace(gridw / 2. * -1, gridw / 2., int(gridw/dy+1))

    if ('x' not in df_pnt.columns)&('y' not in df_pnt.columns):
        df_pnt['x']=df_pnt.geometry.x
        df_pnt['y']=df_pnt.geometry.y
    #creation of transverse vectors    
    Vtmpx=np.divide(\
          (df_pnt['x'].shift(0)-df_pnt['x'].shift(-2))+\
        2*(df_pnt['x'].shift(1)-df_pnt['x'].shift(-1))+\
          (df_pnt['x'].shift(2)-df_pnt['x'].shift(0)),4)
    Vtmpy=np.divide(\
          (df_pnt['y'].shift(0)-df_pnt['y'].shift(-2))+\
        2*(df_pnt['y'].shift(1)-df_pnt['y'].shift(-1))+\
          (df_pnt['y'].shift(2)-df_pnt['y'].shift(0)),4)
    VL=np.sqrt(Vtmpx**2+Vtmpy**2)
    Vox=np.tile(np.divide(Vtmpy,VL)*-1,(len(dL),1))
    Voy=np.tile(np.divide(Vtmpx,VL),(len(dL),1))
    
    dL2=np.tile(np.array(dL).T,(len(Vtmpy),1)).T
    midX=np.tile(df_pnt['x'],(len(dL),1))
    midY=np.tile(df_pnt['y'],(len(dL),1))
    rivkm=np.tile(df_pnt['rkm'],(len(dL),1))
    if 'dist' in df_pnt.columns:
        cdist=np.tile(df_pnt['dist'],(len(dL),1))
    else:
        cdist=np.zeros(rivkm.shape)

    X1=Vox*dL2+midX
    Y1=Voy*dL2+midY
    
    if (feattype=='Point'):
        n=np.tile(range(1,len(dL)+1),(len(Vtmpy),1)).T
        m=np.tile(range(1,len(Vtmpy)+1),(len(dL),1))
        gdf_feat=gpd.GeoDataFrame({'rkm':rivkm.flatten(),'dist':cdist.flatten,'midX':midX.flatten(),'midY':midY.flatten(),'m':m.flatten(),'n':n.flatten()},index=df_pnt.index, geometry=gpd.points_from_xy(X1.flatten(),Y1.flatten()))
    elif (feattype=='LineString'):
        n=np.tile(range(1,len(dL)),(len(Vtmpy),1)).T
        m=np.tile(range(1,len(Vtmpy)+1),(len(dL),1))
        gdf_feat=gpd.GeoDataFrame({'rkm':rivkm[0,:],'dist':cdist[0,:],'midX':midX[0,:],'midY':midY[0,:],'m':m[0,:],'n':n[0,:]},index=df_pnt.index, geometry=[LineString([(row[0],row[2]),(row[1],row[3])]) for row in np.concatenate((X1, Y1),axis=0).T])
    elif (feattype=='Polygon'):   
        def polygon_data(data_array):
            return np.divide(data_array[:-1,:-1]+data_array[1:,:-1]+data_array[1:,1:]+data_array[:-1,1:],4).flatten()
        
        n=np.tile(range(1,len(dL)),(len(Vtmpy)-1,1)).T
        m=np.tile(range(1,len(Vtmpy)),(len(dL)-1,1))
        df_tmp=pd.DataFrame({'rkm':np.round(polygon_data(rivkm)),'dist':polygon_data(cdist),'midX':polygon_data(midX),'midY':polygon_data(midY),'m':m.flatten(),'n':n.flatten(),\
                             'X1':X1[:-1,:-1].flatten(),'X2':X1[1:,:-1].flatten(),'X3':X1[1:,1:].flatten(),'X4':X1[:-1,1:].flatten(),'Y1':Y1[:-1,:-1].flatten(),'Y2':Y1[1:,:-1].flatten(),'Y3':Y1[1:,1:].flatten(),'Y4':Y1[:-1,1:].flatten()},index=df_pnt.index[:-1]).dropna(axis=0)
        gdf_feat=gpd.GeoDataFrame(df_tmp.drop(columns=['X1','X2','X3','X4','Y1','Y2','Y3','Y4']),index=df_tmp.index, geometry=[Polygon([(row['X1'],row['Y1']),(row['X2'],row['Y2']),(row['X3'],row['Y3']),(row['X4'],row['Y4'])]) for index,row in df_tmp.iterrows()])
    else:
        ValueError('This type of feature is not supported (yet)')
  
    return gdf_feat.loc[gdf_feat.index[gdf_feat.is_valid],:]

def point_at_intersect(gdf_cross,gdf_line):
    gdf_cross=check_gdf(gdf_cross)
    gdf_line=check_gdf(gdf_line)
    
    df_points=pd.DataFrame([],columns=[gdf_line.index],index=gdf_cross.index,dtype='object')
    for line,linerow in gdf_line.iterrows():     
        geolist=[]
        for index,row in gdf_cross.iterrows():
            interpoint=row.geometry.intersection(linerow.geometry)
            if interpoint.geom_type=='Point':
                geolist.append(interpoint)
            else:
                nearest_geoms = nearest_points(Point(row['midX'],row['midY']), interpoint)
                geolist.append(nearest_geoms[1])
        # df_points[line]=gpd.GeoSeries([row[1].geometry.intersection(linerow.geometry) for row in gdf_cross.iterrows()])
        workpath=os.getenv('surfdrive') + "\\Documents\\DATA\\Netherlands\\main channel\\main_channelbedlevel\\bed_level_data\\"

        gpd.GeoDataFrame(gdf_cross.rkm,geometry=geolist).to_file(os.path.join(workpath,'worktmp','RL_' + str(line) + '.shp'))
        df_points[line]=gpd.GeoSeries(geolist,index=gdf_cross.index)
        
    return df_points

def create_crossfeature_border(df_pnt,gdf_line,maxW=1e4,frac=None,buffer=None,feattype='Polygon'):
    if ((frac is None)&(buffer is None))|((frac is not None)&(buffer is not None)):
        ValueError('frac or buffer needs to be defined is equal set frac=1')
    
    def compute_shiftpnt(df_dent,gdf_pntline):

        gdf_pntline['x']=gdf_pntline.geometry.x
        gdf_pntline['y']=gdf_pntline.geometry.y
        df_dent['x']=df_dent.geometry.x
        df_dent['y']=df_dent.geometry.y
        
        Vox=gdf_pntline['x']-df_dent['x']
        Voy=gdf_pntline['y']-df_dent['y']
        dL=np.sqrt(Vox**2+Voy**2)
        
        Vox=np.divide(Vox,dL)
        Voy=np.divide(Voy,dL)

        if buffer:
            dL=dL+buffer
        elif frac:
            dL=dL*frac
        
        X1=Vox*dL+df_dent['x']
        Y1=Voy*dL+df_dent['y']
        return gpd.GeoDataFrame({'X':X1,'Y':Y1,'rkm':df_dent.rkm,'dist':df_dent.dist,'midX':df_dent['x'],'midY':df_dent['y'],'m':(df_dent.index-df_dent.index.min())},index=df_dent.index,geometry=[Point(x, y) for x, y in zip(X1, Y1)])

    # create large crosssections
    df_cross=create_crossfeature(df_pnt,maxW,feattype='LineString')
    df_points=point_at_intersect(df_cross,gdf_line)
    
    gdf_R=compute_shiftpnt(df_pnt,gpd.GeoDataFrame([],index=df_points.index,geometry=df_points.iloc[:,0]))
    gdf_L=compute_shiftpnt(df_pnt,gpd.GeoDataFrame([],index=df_points.index,geometry=df_points.iloc[:,1]))
    
    if (feattype=='Point'):
        gdf_R['n']=2
        gdf_L['n']=1
        gdf_feat=gpd.GeoDataFrame(pd.concat([gdf_R,gdf_L]))
    elif (feattype=='LineString'):
        
        gdf_R['XL']=gdf_L.geometry.x
        gdf_R['YL']=gdf_L.geometry.y
        gdf_R['n']=1
        gdf_feat=gpd.GeoDataFrame(gdf_R[gdf_R.columns.difference(['X','Y','XL','YL'])],index=df_pnt.index, geometry=[LineString([(row['X'],row['Y']),(row['XL'],row['YL'])]) for index,row in gdf_R.iterrows()])
                                                                                                                 
    elif (feattype=='Polygon'):
        gdf_R['n']=1
        gdf_R['X2']=gdf_L.geometry.x
        gdf_R['X3']=gdf_L.shift(1).geometry.x
        gdf_R['X4']=gdf_R.shift(1).geometry.x
        
        gdf_R['Y2']=gdf_L.geometry.y
        gdf_R['Y3']=gdf_L.shift(1).geometry.y
        gdf_R['Y4']=gdf_R.shift(1).geometry.y
        
        gdf_R.rkm=np.divide(gdf_R.rkm+gdf_R.rkm.shift(1),2)
        gdf_R.midX=np.divide(gdf_R.midX+gdf_R.midX.shift(1),2)
        gdf_R.midY=np.divide(gdf_R.midY+gdf_R.midY.shift(1),2)
        
        gdf_R.dropna(axis=0,inplace=True)
        gdf_feat=gpd.GeoDataFrame(gdf_R[gdf_R.columns.difference(['X','X2','X3','X4','Y','Y2','Y3','Y4'])],index=gdf_R.index, geometry=[Polygon([(row['X'],row['Y']),(row['X2'],row['Y2']),(row['X3'],row['Y3']),(row['X4'],row['Y4'])]) for index,row in gdf_R.iterrows()])
    else:
        ValueError('This type of feature is not supported (yet)')
  
    return gdf_feat.loc[gdf_feat.is_valid,:]   
    

