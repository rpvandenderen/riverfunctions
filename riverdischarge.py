import numpy as np
import pandas as pd
import matplotlib.dates as mdates

def load_discharge(filename,location='Tiel Waal',sdate=0, edate=1E999,fillgaps=False):
    
    # filename is csv from waterinfo
    # locatgion is measuriong location
    # sdate and edate are start and enddate of the wanted range in matplotlib datenum
    # fillgaps True if you want gaps to be filled by checking other measuring stations
    
    
    df_Q=pd.read_csv(filename,sep=';',decimal=',',usecols=['MEETPUNT_IDENTIFICATIE','WAARNEMINGDATUM', 'WAARNEMINGTIJD','NUMERIEKEWAARDE'],dtype={'MEETPUNT_IDENTIFICATIE':str,'WAARNEMINGDATUM':str,'WAARNEMINGTIJD':str,'NUMERIEKEWAARDE':float})
    
    df_Q['ndate']=np.floor(mdates.date2num(pd.to_datetime(df_Q.iloc[:,1] + " " + df_Q.iloc[:,2],format='%d-%m-%Y %H:%M:%S')))
    idx=(df_Q['ndate']>=sdate)&(df_Q['ndate']<=edate)
    df_Q=df_Q.loc[idx]
    df_Q['NUMERIEKEWAARDE'].loc[df_Q['NUMERIEKEWAARDE']>1E8]=np.nan
    df_Q.dropna(axis=0,inplace=True)
    
    if (location=='Tiel Waal'):
        idx=df_Q.MEETPUNT_IDENTIFICATIE==location
        Qval=df_Q.loc[idx].groupby('ndate').mean()
        
        if fillgaps:
            idx=df_Q.MEETPUNT_IDENTIFICATIE=='Lobith'
            Qval_tmp=df_Q.loc[idx].groupby('ndate').mean()
            Qval=Qval.combine_first(2./3.*Qval_tmp)
    elif (location=='Lobith'):
        idx=df_Q.MEETPUNT_IDENTIFICATIE==location
        Qval=df_Q.loc[idx].groupby('ndate').mean()
        
        if fillgaps:
            idx=df_Q.MEETPUNT_IDENTIFICATIE=='Tiel Waal'
            Qval_tmp=df_Q.loc[idx].groupby('ndate').mean()
            Qval=Qval.combine_first(3./2.*Qval_tmp)
    
    return Qval.iloc[:,0].values,Qval.index.values

def cat_discharge(dates,Qt,Q,Qedges,stat=np.nanmax):
    Qcat=np.zeros(dates.shape[0])*np.nan
    for index,row in enumerate(dates):
        if not np.isnan(row).any():
            Qcat[index]=next(x for x, val in enumerate(Qedges + [1E999]) if val >= stat(Q[(Qt>=row[0])&(Qt<=row[1])]))
    return Qcat

def catmeas_basedonQ(dfdate,datecol,Qdate,Q,Qedges):
    # Qval=pd.DataFrame({'date':Qdate,'Q':Q})
    
    dt=dfdate.pivot_table(index=['m','n'],columns='redname',values=datecol)
    dcat=dt*np.nan
    for index,row in dt.iterrows():
        sertmp=row.dropna().sort_values()
        dcat.at[index,sertmp.index]=cat_discharge(np.stack([sertmp.shift(1).values,sertmp.values],axis=1),Qdate,Q,Qedges)
    # def merge_dfdate(r):
    #     return dcat.loc[(r.m,r.n),r.redname]
    dfnew=pd.merge(dfdate,pd.DataFrame(dcat.stack()).rename(columns={0:'Qcat'}), how='inner',left_on=['m','n','redname'],right_on=['m','n','redname'])
    return dfnew['Qcat']