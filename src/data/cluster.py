import numpy as np
from pylorentz import Momentum4

from .position import Position

class Cluster:
  def __init__ (self, cluster, fields):
    for name, python_name in fields:
      setattr(self, python_name, cluster[name])
    
  def has_field (self, field):
    return hasattr(self, field)

  def position (self):
    return Position(self.cal_eta, self.cal_phi)
  
  def momentum (self):
    return Momentum4.e_m_eta_phi(self.cal_e, 0, self.cal_eta, self.cal_phi)
  