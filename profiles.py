import math
import numpy as np
from scipy import integrate
from scipy.stats import norm
from matplotlib import pyplot as plt
import random
import itertools
import copy

tau = 2 * math.pi
pi = math.pi

class Panel:
    def __init__(self, x_a, y_a, x_b, y_b):
        self.x_a, self.y_a = x_a, y_a
        self.x_b, self.y_b = x_b, y_b
        self.x_c, self.y_c = (x_a + x_b) * 0.5, (y_a + y_b) * 0.5
        self.length = ((x_b - x_a)**2 + (y_b - y_a)**2)**0.5
        
        if x_b <= x_a:
            self.beta = math.acos((y_b - y_a) / self.length)
        else:
            self.beta = pi + math.acos(-(y_b - y_a) / self.length)
            
        if self.beta <= pi:
            self.loc = "upper"
        else:
            self.loc = "lower"
            
        if self.y_a < self.y_b:
            self.grad = "positive"
        elif self.y_a > self.y_b:
            self.grad = "negative"
        else:
            self.grad = "neutral"
            
        self.sigma = 0
        self.v_t = 0
        self.cp = 0
        
def define_panels(x, y, N = 50):
    R = (x.max() - x.min()) / 2
    x_center = (x.max() + x.min()) / 2
    x_circle = x_center + R * np.cos(np.linspace(0, tau, N + 1))
    
    x_ends = np.copy(x_circle)
    y_ends = np.empty_like(x_ends)
    
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    I=0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x [I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a*x_ends[i] + b
    y_ends[N] = y_ends[0]
    
    panels = np.empty(N, dtype = object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i+1], y_ends[i+1])
        
    return panels

class Freestream:
    def __init__(self, u_inf = 1, alpha = 0):
        self.u_inf = u_inf
        self.alpha = np.radians(alpha)
        
def integral(x, y, panel, dxdz, dydz):
    """
    oblicza prędkość w (x, y) generowaną przez panel panel w kierunku z
    """
    def integrand(s):
        x_s = panel.x_a - math.sin(panel.beta) * s
        y_s = panel.y_a + math.cos(panel.beta) * s
        
        numerator = (x - x_s) * dxdz + (y - y_s) * dydz
        denominator = (x - x_s)**2 + (y - y_s)**2
        return numerator / denominator
    return integrate.quad(integrand, 0, panel.length)[0]

def source_contribution_normal(panels):
    """
    zwraca macierz źródeł prędkości normalnej dla każdego panelu
    """
    A = np.empty((panels.size, panels.size), dtype = float)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = integral(panel_i.x_c, panel_i.y_c, panel_j, np.cos(panel_i.beta), np.sin(panel_i.beta)) / tau
            else:
                A[i, j] = 0.5
    return A

def vortex_contribution_normal(panels):
    """
    zwraca macierz wirów prędkości normalnej dla każdego panelu
    """
    A = np.empty((panels.size, panels.size), dtype = float)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -integral(panel_i.x_c, panel_i.y_c, panel_j, np.sin(panel_i.beta), -np.cos(panel_i.beta)) / tau
            else:
                A[i, j] = 0
    return A

def kutta_cond(A_source, B_vortex):
    """
    zwraca macierz warunku Kutty-Żukowskiego dla każdego panelu
    """
    b = np.empty(A_source.shape[0] + 1, dtype = float)
    b[:-1] =  B_vortex[0, :] + B_vortex[-1, :]
    b[-1] = -np.sum(A_source[0, :] + A_source[-1, :])
    return b

def build_singularity_matrix(A_source, B_vortex):
    """
    zwraca macierz lewych stron układu równań wyznaczonego przez istnienie źródeł A i wirów B
    """
    A = np.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype = float)
    A[:-1, :-1] = A_source
    A[:-1, -1] = np.sum(B_vortex, axis = 1)
    A[-1, :] = kutta_cond(A_source, B_vortex)
    return A

def build_freestream_rhs(panels, freestream):
    """
    zwraca wektor prawych stron układu wynikającego z przepływu niezaburzonego
    """
    b = np.empty(panels.size + 1, dtype = float)
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.alpha - panel.beta)
        b[-1] = -freestream.u_inf * (np.sin(freestream.alpha - panels[0].beta) + np.sin(freestream.alpha - panels[-1].beta))
    return b

def compute_v_t(panels, freestream, gamma, A_source, B_vortex):
    """
    wylicza wektor prędkości stycznych paneli przy zadanej cyrkulacji, przepływie niezaburzonym
    i macierzy źródeł i wirów paneli.
    Przypisuje ten wektor panelom.
    """
    A = np.empty((panels.size, panels.size + 1), dtype = float)
    A[:, :-1] = B_vortex
    A[:, -1] = -np.sum(A_source, axis = 1)
    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta for panel in panels])
    strengths = np.append([panel.sigma for panel in panels], gamma)
    v_t = np.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.v_t = v_t[i]
        
def compute_cp(panels, freestream):
    """
    wylicza współczynnik ciśnienia dla każdego panelu,
    przy zadanym przepływie niezaburzonym i znanych prędkościach stycznych
    """
    for panel in panels:
        panel.cp = 1 - (panel.v_t / freestream.u_inf)**2
        
class Profile:
    
    last_ID = 0

    def __init__(self, x, y, N = 50, ID = None):
        self.x = x
        self.y = y
        self.N = N
        self.panels = define_panels(self.x, self.y, self.N)
        self.chord = abs(max(panel.x_a for panel in self.panels) - min(panel.x_a for panel in self.panels))
        
        if ID is not None:
            self.ID = ID
        else:
            Profile.last_ID += 1
            self.ID = Profile.last_ID
    
    def copy(self):
        new_instance = copy.deepcopy(self)
        Profile.last_ID += 1
        new_instance.ID = Profile.last_ID
        return new_instance
            
        
        
    def get_aero(self, freestream):
        self.freestream = freestream
        M_source = source_contribution_normal(self.panels)
        M_vortex = vortex_contribution_normal(self.panels)
        
        A = build_singularity_matrix(M_source, M_vortex)
        b = build_freestream_rhs(self.panels, self.freestream)
        
        strengths = np.linalg.solve(A, b)
        self.gamma = strengths[-1]
        for i, panel in enumerate(self.panels):
            panel.sigma = strengths[i]
            
        compute_v_t(self.panels, self.freestream, self.gamma, M_source, M_vortex)
        compute_cp(self.panels, self.freestream)
        self.cl = self.gamma * sum(panel.length for panel in self.panels) / (0.5 * self.freestream.u_inf * self.chord)
        
        self.accuracy = sum([panel.sigma * panel.length for panel in self.panels])
        
    def draw(self, width = 10, height = 10):
        plt.figure(figsize = (width, height))
        plt.grid()
        plt.xlabel("x", fontsize = 15)
        plt.ylabel("y", fontsize = 15)
        plt.plot(self.x, self.y)
        plt.axis("scaled")
        plt.plot(np.append([panel.x_a for panel in self.panels], self.panels[0].x_a),
         np.append([panel.y_a for panel in self.panels], self.panels[0].y_a),
        marker = "o", color = "red", linewidth = 0.5)
        
    def dent(self, mu, d):
        x0 = min(self.x, key = lambda x: abs(x - mu))
        difference = norm.pdf(self.x, x0) * d
        self.y = np.array([sum(yi) for yi in zip(self.y, difference)])
        self.panels = define_panels(self.x, self.y, self.N)
        
    def blow(self, mu, d):
        """
        WIP - not used
        """
        x0 = min(self.x, key = lambda x: abs(x - mu))
        difference = norm.pdf(self.x, x0) * d
        self.y = [(y + y_d) if  panel.loc == "upper" else (y - y_d) for (y, panel, y_d) in zip(self.y, self.panels, difference)]      
        
    def draw_cp(self):
        width, height = 10, 10
        plt.figure(figsize = (width, height))
        plt.grid()
        plt.xlabel("x", fontsize = 20)
        plt.ylabel("$C_p$", fontsize = 20)
        plt.plot([p.x_c for p in self.panels if p.loc == "upper"],
                 [p.cp for p in self.panels if p.loc == "upper"],
                 label = "upper", color = "red", linewidth = 0.5, marker = "X", markersize = 10)
        plt.plot([p.x_c for p in self.panels if p.loc == "lower"],
                 [p.cp for p in self.panels if p.loc == "lower"],
                 label = "lower", color = "blue", linewidth = 0.5, marker = "o", markersize = 10)
        plt.legend(loc = "best", prop = {"size": 20})
        plt.xlim(-0.1, 1.1)
        plt.ylim(min([p.cp for p in self.panels]) - 1, max([p.cp for p in self.panels]) + 1)
        
    def get_limitations(self):
        """
        shadow - vertical shadow (span - perpendicular to chord) of the profile - emulates drag
        roughness - number of local extrema on the profile's upper curve
        """
        self.shadow = abs(max(panel.y_a for panel in self.panels) - min(panel.y_a for panel in self.panels))
        
        self.roughness = 0
        prev_grad = "neutral"
        for p in [p for p in self.panels if p.loc == "upper"]:
            if prev_grad != "neutral" and prev_grad != p.grad:
                self.roughness += 1
            prev_grad = p.grad

