# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import math

class Chan_VeseModel():
    
    def __init__(self, img):
        # Chan-Vese参数
        self.__mu = 0.5
        self.__upsilon = 0
        self.__lambda1 = 1
        self.__lambda2 = 1
        self.__dt = 0.1
        self.__h = 1
        self.__p = 1
        self.__maxIters = 10
        
        self.__img = img
        self.__phis = {}
        self.__C1C2 = {}
        
    def _initPHI(self):
        width, height = self.__img.size
        phi = np.zeros( (height, width), dtype=np.float64 )
        x_center = int(width / 2)
        y_center = int(height / 2)
        radius = 0.1 * min(height, width)
        for y in range(height):
            for x in range(width):
                distance_2 = math.pow(x - x_center, 2) + math.pow(y - y_center, 2)
                phi[y, x] = (math.pow(radius, 2) - distance_2) / (math.pow(radius, 2) + distance_2)  
        return phi      
    
    def _showPHI(self, phi, isShow):
        height = phi.shape[0]
        width = phi.shape[1]
        phi_show = np.zeros( (height, width), dtype=np.uint8)
        if isShow:
            img_show = self.__img.copy()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (0 == phi[y, x]):
                    if (0 != phi[y - 1, x - 1] or 
                        0 != phi[y - 1, x] or 
                        0 != phi[y - 1, x + 1] or 
                        0 != phi[y, x - 1] or 
                        0 != phi[y, x + 1] or 
                        0 != phi[y + 1, x - 1] or
                        0 != phi[y + 1, x] or 
                        0 != phi[y + 1, x + 1]):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                else:
                    if ( (abs(phi[y, x]) < abs(phi[y - 1, x - 1])) and (phi[y, x] > 0) != (phi[y - 1, x - 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y - 1, x])) and (phi[y, x] > 0) != (phi[y - 1, x] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y - 1, x + 1])) and (phi[y, x] > 0) != (phi[y - 1, x + 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y, x - 1])) and (phi[y, x] > 0) != (phi[y, x - 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y, x + 1])) and (phi[y, x] > 0) != (phi[y, x + 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y + 1, x - 1])) and (phi[y, x] > 0) != (phi[y + 1, x - 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y + 1, x])) and (phi[y, x] > 0) != (phi[y + 1, x] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
                    if ( (abs(phi[y, x]) < abs(phi[y + 1, x + 1])) and (phi[y, x] > 0) != (phi[y + 1, x + 1] > 0) ):
                        phi_show[y, x] = 255
                        if isShow:
                            img_show.putpixel((x, y), 255)
          
        phi_img = Image.fromarray(phi_show)
        phi_img.show()
        if isShow:
            img_show.show()
        
    def _computeC1C2(self, i):
        if i in self.__C1C2: print("The value of C1 and C2 have been exist!"); return
        print("\tCompute mean values for region inner and outer the phi...")
        phi = self.__phis[i]
        width, height = self.__img.size
        assert(phi.shape[0] == height and phi.shape[1] == width)
        C1 = 0; C2 = 0
        N1 = 0; N2 = 0
        for y in range(height):
            for x in range(width):
                pixel = phi[y, x]
                if pixel >= 0:
                    C1 += self.__img.getpixel((x, y))
                    N1 += 1
                else:
                    C2 += self.__img.getpixel((x, y))
                    N2 += 1
        C1 = C1 / N1
        C2 = C2 / N2
        self.__C1C2[i] = (C1, C2)
        print("\tC1 C2 Computation done, C1 = %f, C2 = %f"%(C1, C2))
        
    def _Dirac(self, z, epsilon = 1.0):
        return 1 / np.pi * epsilon / ( np.power(epsilon, 2) + np.power(z, 2) )
    
    def _PDESolver(self, i):
        ''' 
        偏微分方程求解器
            输入: i, 迭代索引
            输出：更新的水平集
        '''
        
        print("\tSolve PDE for curve evoluation...")
        phi = self.__phis[i]
        phiNext = np.zeros( (phi.shape[0], phi.shape[1]), dtype=np.float64 )
        meanC1 = self.__C1C2[i][0]
        meanC2 = self.__C1C2[i][1]
        eps = 1e-6
        for y in range(1, phi.shape[0] - 1):
            for x in range(1, phi.shape[1] - 1):
                C1 = 1 / np.sqrt( eps + np.power((phi[y + 1, x] - phi[y, x]), 2) + np.power((phi[y, x + 1] - phi[y, x - 1]), 2) / 4 )
                C2 = 1 / np.sqrt( eps + np.power((phi[y, x] - phi[y - 1, x]), 2) + np.power((phi[y - 1, x + 1] - phi[y - 1, x - 1]), 2) / 4 )
                C3 = 1 / np.sqrt( eps + np.power((phi[y + 1, x] - phi[y - 1, x]), 2) / 4 + np.power((phi[y, x + 1] - phi[y, x]), 2) )
                C4 = 1 / np.sqrt( eps + np.power((phi[y + 1, x - 1] - phi[y - 1, x - 1]), 2) / 4 + np.power((phi[y, x] - phi[y, x - 1]), 2) )
                C = C1 + C2 + C3 + C4
                
                deltaPhi = self._Dirac(phi[y, x])
                factor = self.__dt * deltaPhi * self.__mu
                F1 = factor * C1 / (self.__h + factor * C)
                F2 = factor * C2 / (self.__h + factor * C)
                F3 = factor * C3 / (self.__h + factor * C)
                F4 = factor * C4 / (self.__h + factor * C)
                F = self.__h / (self.__h + factor * C)
                pij = phi[y, x] - self.__dt * deltaPhi * ( self.__upsilon + self.__lambda1 * np.power(self.__img.getpixel((x, y)) - meanC1, 2) - self.__lambda2 * np.power(self.__img.getpixel((x, y)) - meanC2, 2) )
                phiNext[y, x] = F1 * phi[y + 1, x] + F2 * phi[y - 1, x] + F3 * phi[y, x + 1] + F4 * phi[y, x - 1] + F * pij
                
        for y in range(phiNext.shape[0]):
            phiNext[y, 0] = phiNext[y, 1]
            phiNext[y, phiNext.shape[1] - 1] = phiNext[y, phiNext.shape[1] - 2]
        for x in range(phiNext.shape[1]):
            phiNext[0, x] = phiNext[1, x]
            phiNext[phiNext.shape[0] - 1, x] = phiNext[phiNext.shape[0] - 2, x]
            
        print("\tPDE solved.\n")
        return phiNext
        
    def _checkStop(self, phi, phiNext):
        L = 0
        M = 0
        assert(phi.shape[0] == phiNext.shape[0] and phi.shape[1] == phiNext.shape[1])
        height = phi.shape[0]
        width = phi.shape[1]
        for y in range(height):
            for x in range(width):
                if (phi[y, x] < self.__h): M += 1
                L += abs(phiNext[y, x] - phi[y, x])
        Q = 0
        if M > 0: Q = L / M
        if (Q - self.__dt * np.power(self.__h, 2) < 1e-6):
            return True
        else:
            return False     
    
    def _reInitPHI(self, phi, maxIter):
        print("\tReinitialize phi...\n")
        stop = False
        psi = phi
        for i in range(maxIter):
            if stop: 
                break
            psiOld = psi
            for y in range(1, phi.shape[0] - 1):
                for x in range(1, phi.shape[1] - 1):
                    a = (psiOld[y, x] - psiOld[y - 1, x]) / self.__h
                    b = (psiOld[y + 1, x] - psiOld[y, x]) / self.__h
                    c = (psiOld[y, x] - psiOld[y, x - 1]) / self.__h
                    d = (psiOld[y, x + 1] - psiOld[y, x]) / self.__h
                    G = 0
                    if (psiOld[y, x] > 0):
                        G = np.sqrt(max(np.power(max(a, 0.0), 2), np.power(min(b, 0.0), 2)) + max(np.power(max(c, 0.0), 2), np.power(min(d, 0.0), 2))) - 1
                    elif (psiOld[y, x] < 0):
                        G = np.sqrt(max(np.power(min(a, 0.0), 2), np.power(max(b, 0.0), 2)) + max(np.power(min(c, 0.0), 2), np.power(max(d, 0.0), 2))) - 1
                    sign = 1 if (psiOld[y, x] >= 0) else -1
                    psi[y, x] = psiOld[y, x] - self.__dt * sign * G
            stop = self._checkStop(phi, psi)
        print("\tReinitialization done.\n")
        return psi
     
        
    def solver(self):
        # 初始化水平集函数phi
        print("Initialize phi ...")
        phi0 = self._initPHI()
        self.__phis[0] = phi0
        print("Initialization done.")
        self._showPHI(phi0, False)
        for i in range(self.__maxIters):
            print("Iteration %d for Chan-Vese curve evolution..."%(i))
            self._computeC1C2(i)
            phiNew = self._PDESolver(i)
            numIter = 100
            phiNext = self._reInitPHI(phiNew, numIter)
            self.__phis[i + 1] = phiNext
            self._showPHI(phiNext, True)
            print("Iteration %d evolution done."%(i))
            
if __name__ == "__main__":
    img = Image.open("/home/zzx/work/DeepMLG/neuralNets/UNetLevelSet/data/1.jpg").convert("L")
    cv = Chan_VeseModel(img)
    cv.solver()
        
