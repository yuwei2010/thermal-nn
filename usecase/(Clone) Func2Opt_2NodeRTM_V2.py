# 2-Node RTM with 41 Parameters
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,interp2d,RegularGridInterpolator
from scipy.stats import norm
from joblib import Parallel, delayed
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
    
def Func2Opt_2NodeRTM_V2(params, tqEm, nEm, stSwtMod, PvEdyRtrRaw, PvHysRtrRaw, PvPwmRtrRaw, tRotorMagMaxFild, tOilGbxSnsrFild100ms, tCooltIvtrOutl, vfCooltIvtr, ind_ini,flagEnaPlot=0):
    tic()
    Ts = 0.5
    InvRth_Coolnt2StrLUTX_IniInterp = np.array([0, 1.5, 3.5, 8, 10])
    InvRth_Coolnt2StrLUTY_IniInterp = np.array([0.3138, 30.0338, 49.4652, 80.0000, 90.9091])
    InvRth_Rtr2StrLUTX_IniInterp = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12000, 14000])
    InvRth_Rtr2StrLUTY_IniInterp = np.array([1.8111, 13.4566, 15.0198, 15.8411, 16.6624, 17.1839, 17.7054, 18.4091, 19.6065, 20.4679, 21.2030, 21.7338, 22.2647])
    InvRth_Oil2RtrLUTX_IniInterp = np.array([-3500, -2000, -1000, 0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12000, 14000])
    InvRth_Oil2RtrLUTY_IniInterp = np.array([0.3138, 0.3138, 0.3138, 0.3138, 18.8196, 28.0547, 35.9712, 42.2934, 48.1283, 53.4759, 58.1731, 62.8704, 67.5676, 71.7117, 75.8559, 80.0000])
    
    InvRth_Coolnt2StrLUTY_Interp_New = np.array(params[35:40]) * InvRth_Coolnt2StrLUTY_IniInterp
    InvRth_Rtr2StrLUTY_Interp_New = np.array(params[0:13]) * InvRth_Rtr2StrLUTY_IniInterp
    InvRth_Oil2RtrLUTY_Interp_New = np.array(params[13:29]) * InvRth_Oil2RtrLUTY_IniInterp
    
    interpolator_InvRth_Coolnt2Str = interp1d(InvRth_Coolnt2StrLUTX_IniInterp, InvRth_Coolnt2StrLUTY_Interp_New, kind='linear', fill_value='extrapolate')
    interpolator_InvRth_Rtr2Str = interp1d(InvRth_Rtr2StrLUTX_IniInterp, InvRth_Rtr2StrLUTY_Interp_New, kind='linear', fill_value='extrapolate')
    interpolator_InvRth_Oil2Rtr = interp1d(InvRth_Oil2RtrLUTX_IniInterp, InvRth_Oil2RtrLUTY_Interp_New, kind='linear', fill_value='extrapolate')
    
    InvCth_RtrNew = params[29] * 2.9787e-04
    InvCth_StrNew = params[40] * 2.9787e-04
    
    dataLength = len(nEm)
    
    InvRth_Coolnt2Str_temp = interpolator_InvRth_Coolnt2Str(np.maximum(vfCooltIvtr, 0))
    InvRth_Rtr2Str_temp = interpolator_InvRth_Rtr2Str(np.abs(nEm))
    InvRth_Oil2Rtr_temp = interpolator_InvRth_Oil2Rtr(nEm)
    
    pwrLossRtrEstd_temp = PvEdyRtrRaw * params[30] + PvHysRtrRaw * params[31]
    PvPwmRtr_temp = np.copy(PvPwmRtrRaw)
    PvPwmRtr_temp[stSwtMod != 5] *= params[32]
    PvPwmRtr_temp[stSwtMod == 5] *= params[33]
    # PvAddRtr_temp = params[34] * np.abs(nEm) + params[35] * np.abs(nEm)**2 + params[36] * np.abs(nEm)**3
    PvAddRtr_temp = params[34] * (17.56 + 0.13*np.abs(tqEm) + 7.174e-6*tqEm**2 + 3.093e-6 *np.abs(tqEm)*nEm**2)
    pwrLossRtrEstd_temp = np.maximum(pwrLossRtrEstd_temp + PvPwmRtr_temp + PvAddRtr_temp, 0)
    
    tmpRtrEstd = np.empty(dataLength)
    tmpStrEstd = np.empty(dataLength)
    
    tmpRtrEstd[0] = tRotorMagMaxFild[0]
    tmpStrEstd[0] = (tRotorMagMaxFild[0] + tCooltIvtrOutl[0]) / 2
    
    for k in range(0, dataLength - 1):
        if k in ind_ini:
            tmpRtrEstd[k] = tRotorMagMaxFild[k]
            tmpStrEstd[k] = (tRotorMagMaxFild[k] + tCooltIvtrOutl[k]) / 2
        tmpStrEstd[k + 1] = Ts * ((tmpRtrEstd[k] - tmpStrEstd[k]) * InvCth_StrNew * InvRth_Rtr2Str_temp[k] + (tCooltIvtrOutl[k] - tmpStrEstd[k]) * InvCth_StrNew * InvRth_Coolnt2Str_temp[k]) + tmpStrEstd[k]
        tmpRtrEstd[k + 1] = Ts * ((tmpStrEstd[k] - tmpRtrEstd[k]) * InvCth_RtrNew * InvRth_Rtr2Str_temp[k] + (tOilGbxSnsrFild100ms[k] - tmpRtrEstd[k]) * InvCth_RtrNew * InvRth_Oil2Rtr_temp[k] + pwrLossRtrEstd_temp[k] * InvCth_RtrNew) + tmpRtrEstd[k]
        if np.isnan(tmpRtrEstd[k + 1]):
            return float('inf')
    
    total_error = np.sum((tmpRtrEstd - tRotorMagMaxFild) ** 2) / dataLength
    if np.isnan(total_error):
        total_error = float('inf')
    
    if flagEnaPlot == 1:
        tmpRtrEstd_acum = tmpRtrEstd
        tRotorMaxRef_acum = tRotorMagMaxFild
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
        plt.savefig('tRotorEstd_vs_tRotorMeasured_2NodeRTM_V2.png')
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
        plt.savefig('Histogram_2NodeRTM_V2.png')
        plt.close()
    
    toc()
    return total_error

# results = Func2Opt_1NodeRTM(result_1NodeRTM.x,measurements)

