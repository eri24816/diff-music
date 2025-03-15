import numpy as np



class InformationCalculator:
    def __init__(self, px0y0, px0y1, px1y0, px1y1):
        self.px0y0 = px0y0
        self.px0y1 = px0y1
        self.px1y0 = px1y0
        self.px1y1 = px1y1
        self.px0 = px0y0 + px0y1
        self.px1 = px1y0 + px1y1
        self.py0 = px0y0 + px1y0
        self.py1 = px0y1 + px1y1

    def h_x(self):
        return -self.px0*np.log2(self.px0)-self.px1*np.log2(self.px1)

    def h_y(self):
        return -self.py0*np.log2(self.py0)-self.py1*np.log2(self.py1)

    def h_xy(self):
        return -self.px0y0*np.log2(self.px0y0)-self.px0y1*np.log2(self.px0y1)-self.px1y0*np.log2(self.px1y0)-self.px1y1*np.log2(self.px1y1)

    def i_xy(self):
        return self.h_x()+self.h_y()-self.h_xy()
    
    def h_x_given_y(self):
        return self.h_xy()-self.h_y()

    def h_y_given_x(self):
        return self.h_xy()-self.h_x()
    
