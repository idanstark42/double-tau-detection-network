import numpy as np
from pylorentz import Momentum4

from .position import Position

class Truth:
  def __init__ (self, truth, fields):
    for name, python_name in fields:
      setattr(self, python_name, truth[name])

  def visible_position (self):
    return Position(self.eta_vis, self.phi_vis)
  
  def invisible_position (self):
    return Position(self.eta_invis, self.phi_invis)
  
  def visible_momentum (self):
    return Momentum4.m_eta_phi_pt(self.pt_vis, self.eta_vis, self.phi_vis, self.m_vis)
  
  def invisible_momentum (self):
    return Momentum4.m_eta_phi_pt(self.pt_invis, self.eta_invis, self.phi_invis, self.m_invis)
  
  def total_momentum (self):
    return self.visible_momentum() + self.invisible_momentum()
