# 1Node RTM with 33 Parameters 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,interp2d,RegularGridInterpolator
from scipy.stats import norm
import numpy as np
import time
import pickle
# Simulate the tic/toc function as in Matlab
def tic():
    global start_time
    start_time = time.time()

# Simulate the tic/toc function as in Matlab
def toc():
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
def Func2Opt_1NodeRTM_V4(params,measurements,flagEnaPlot=0):
    tic()
    total_error = 0 
    Ts=0.5
    InvRth_Coolnt2RtrLUTX_IniInterp = np.array([0,1.5,3.5,8,10]) # water coolant flowrate
    InvRth_Coolnt2RtrLUTY_IniInterp = np.array([0.3138,30.0338,49.4652,80.0000,90.9091]) 
    # InvRth_Rtr2StrLUTX_IniInterp = np.array([0,500,1000,1500,2000,2500,3000,4000,6000,8000,10000,12000,14000]) # rotor speed
    # InvRth_Rtr2StrLUTY_IniInterp = np.array([1.8111,13.4566,15.0198,15.8411,16.6624,17.1839,17.7054,18.4091,19.6065,20.4679,21.2030,21.7338,22.2647])
    InvRth_Oil2RtrLUTX_IniInterp = np.array([-3500,-2000,-1000,0,500,1000,1500,2000,2500,3000,4000,6000,8000,10000,12000,14000])   # rotor speed
    InvRth_Oil2RtrLUTY_IniInterp = np.array([0.3138,0.3138,0.3138,0.3138,18.8196,28.0547,35.9712,42.2934,48.1283,53.4759,58.1731,62.8704,67.5676,71.7117,75.8559,80.0000])
    InvRth_Rtr2StrLUTY_Interp_New = np.array(params[0:5])*InvRth_Coolnt2RtrLUTY_IniInterp
    InvRth_Oil2RtrLUTY_Interp_New = np.array(params[5:21])*InvRth_Oil2RtrLUTY_IniInterp
    interpolator_InvRth_Rtr2Str=interp1d(InvRth_Coolnt2RtrLUTX_IniInterp,InvRth_Rtr2StrLUTY_Interp_New,kind ='linear',fill_value='extrapolate')
    interpolator_InvRth_Oil2Rtr=interp1d(InvRth_Oil2RtrLUTX_IniInterp,InvRth_Oil2RtrLUTY_Interp_New,kind ='linear',fill_value='extrapolate')
    InvCth_RtrNew = params[21]*2.9787e-04
    tmpRtrEstd_acum=[]
    tRotorMaxRef_acum=[]
    for tqEm,nEm,stSwtMod,PvCuDcRaw,PvCuAcRaw,PvEdySttrRaw,PvHysSttrRaw,PvEdyRtrRaw,PvHysRtrRaw,PvPwmRtrRaw,tRotorMagMaxFild,tOilGbxSnsrFild100ms,tCooltIvtrOutl,vfCooltIvtr in measurements:
        InvRth_Rtr2Str_temp = interpolator_InvRth_Rtr2Str(np.maximum(vfCooltIvtr,0))
        InvRth_Rtr2Str_temp = np.maximum((((tCooltIvtrOutl-65)*params[31] + 1)**params[32]),0.1)*InvRth_Rtr2Str_temp
        InvRth_Oil2Rtr_temp=interpolator_InvRth_Oil2Rtr(nEm)
        PvPwmRtr_temp = np.copy(PvPwmRtrRaw)
        PvPwmRtr_temp[stSwtMod != 5] *= params[22]
        PvPwmRtr_temp[stSwtMod == 5] *= params[23]  
        pwrLossRtrEstd_temp = np.maximum(PvCuDcRaw*params[24] + PvCuAcRaw*params[25] + PvEdySttrRaw*params[26] + PvHysSttrRaw*params[27] + PvEdyRtrRaw*params[28] + PvHysRtrRaw*params[29] + PvPwmRtr_temp + params[30]*(17.56 + 0.13*np.abs(nEm) + 7.172e-6*nEm**2 + 3.093e-6 *np.abs(nEm)*tqEm**2),0) 
        tmpRtrEstd=[]
        for k in range(0,len(nEm)-1):
            if k==0:
                # tmpRtrEstd[k] = tRotorMagMaxFild[k]
                tmpRtrEstd = np.append(tmpRtrEstd,[tRotorMagMaxFild[k]],axis=0)
            tmpRtrEstd_NextStep = Ts*((tCooltIvtrOutl[k]- tmpRtrEstd[k])*InvCth_RtrNew*InvRth_Rtr2Str_temp[k]+(tOilGbxSnsrFild100ms[k]- tmpRtrEstd[k])*InvCth_RtrNew*InvRth_Oil2Rtr_temp[k]+ pwrLossRtrEstd_temp[k]*InvCth_RtrNew) + tmpRtrEstd[k]   
            if np.isnan(tmpRtrEstd_NextStep): 
                return float('inf')
            tmpRtrEstd = np.append(tmpRtrEstd,[tmpRtrEstd_NextStep],axis=0)
        if flagEnaPlot == 1:
            tmpRtrEstd_acum = np.append(tmpRtrEstd_acum,tmpRtrEstd,axis=0)
            tRotorMaxRef_acum = np.append(tRotorMaxRef_acum,tRotorMagMaxFild,axis=0)

        total_error += np.sum((tmpRtrEstd - tRotorMagMaxFild) ** 2)
    if flagEnaPlot == 1:
        t = np.arange(0, len(tmpRtrEstd_acum)*0.5, 0.5)
        fig1 = plt.figure(figsize=(8,5))
        plt.plot(t,tmpRtrEstd_acum,label='estimated')
        plt.plot(t,tRotorMaxRef_acum,label='measured')
        plt.plot(t,tmpRtrEstd_acum-tRotorMaxRef_acum,label='Diff')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        plt.title('tRotorEstd VS tRotorMeasured')
        # plt.show()
        # 保存图形到文件
        plt.savefig('tRotorEstd_vs_tRotorMeasured_1NodeRTM_V4.png')
        plt.close()
        mu, std = norm.fit(tmpRtrEstd_acum-tRotorMaxRef_acum)
        fig2 = plt.figure(figsize=(8,5))
        # 绘制直方图
        plt.hist(tmpRtrEstd_acum-tRotorMaxRef_acum, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')
        # 绘制拟合的正态分布曲线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        # 标注 mu 所对应的 x 位置
        plt.axvline(mu, color='r', linestyle='dashed', linewidth=2, label=f'Mu = {mu:.2f}')
        # 标注 xmin 和 xmax 的位置
        plt.axvline(mu+3*std, color='orange', linestyle='dotted', linewidth=2, label=f'Mu + 3Std = {mu+3*std:.2f}')
        plt.axvline(mu-3*std, color='orange', linestyle='dotted', linewidth=2, label=f'Mu - 3Std = {mu-3*std:.2f}')
        plt.axvline(xmin, color='b', linestyle='dotted', linewidth=2, label=f'MinErr = {xmin:.2f}')
        plt.axvline(xmax, color='b', linestyle='dotted', linewidth=2, label=f'MaxErr = {xmax:.2f}')
        title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
        plt.title(title)
        plt.xlabel('Estd-Measd (K)')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.legend()
        # plt.show()
        # 保存图形到文件
        plt.savefig('Histogram_1NodeRTM_V4.png')
        plt.close()
   
    if np.isnan(total_error): 
        total_error = float('inf')
    toc()
    return total_error

# results = Func2Opt_1NodeRTM(result_1NodeRTM.x,measurements)

