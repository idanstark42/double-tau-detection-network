import numpy as np
from sklearn.preprocessing import StandardScaler
from pylorentz import Momentum4

from data.cluster import Cluster
from data.track import Track
from data.tau_truth import Truth

from settings import FIELDS_TO_NORMALIZE, CHANNEL_START

class Event:
  def __init__ (self, event, clusters, tracks, truth, event_fields, clusters_fields, tracks_fields, truthTaus_fields, normalize_fields=False, normalize_energy=True):
    self.average_interactions_per_crossing = event[0]
    self.mc_channel_number = event[1] if len(event) > 1 else 0
    self.clusters = [Cluster(cluster, clusters_fields) for cluster in clusters if cluster['valid']]
    self.tracks = [Track(track, tracks_fields) for track in tracks if track['valid']]
    self.truths = [Truth(truth, truthTaus_fields) for truth in truth if truth['valid']]

    # if the clusters are a string, print the string
    if isinstance(clusters, str):
      print(clusters)

    self.clusters = [cluster for cluster in self.clusters if cluster.position().in_range()]
    self.tracks = [track for track in self.tracks if track.position().in_range()]
    self.truths = [truth for truth in self.truths if truth.visible_position().in_range()]

    self.normalize_fields = normalize_fields
    self.normalize_energy = normalize_energy

    self._calculateion_cache = {}
    self.clusters_scaler = StandardScaler()
    self.tracks_scaler = StandardScaler()

    self.normalize()

  def normalize (self):
    if not self.normalize_fields and not self.normalize_energy:
      return
    # normalize clusters
    if self.normalize_fields:
      normalizable_clusters_fields_values = np.array([[getattr(cluster, field) for cluster in self.clusters] for field in FIELDS_TO_NORMALIZE['clusters']]).T
      normalized_cluster_fields_values = self.clusters_scaler.fit_transform(normalizable_clusters_fields_values)
    max_energy = max([cluster.cal_e for cluster in self.clusters])
    for index, cluster in enumerate(self.clusters):
      if self.normalize_energy:
        cluster.cal_e /= max_energy
      if self.normalize_fields:
        for field in FIELDS_TO_NORMALIZE['clusters']:
          setattr(cluster, field, normalized_cluster_fields_values[index][FIELDS_TO_NORMALIZE['clusters'].index(field)])
    
    # normalize tracks
    if self.normalize_fields:
      normalizable_tracks_fields_values = np.array([[getattr(track, field) for track in self.tracks] for field in FIELDS_TO_NORMALIZE['tracks']]).T
      normalized_track_fields_values = self.tracks_scaler.fit_transform(normalizable_tracks_fields_values)
    max_pt = max([track.pt for track in self.tracks])
    for index, track in enumerate(self.tracks):
      if self.normalize_energy:
        track.pt /= max_pt
      if self.normalize_fields:
        for field in FIELDS_TO_NORMALIZE['tracks']:
          setattr(track, field, normalized_track_fields_values[index][FIELDS_TO_NORMALIZE['tracks'].index(field)])

  def calculate_and_cache (self, key, calculation):
    if key not in self._calculateion_cache:
      self._calculateion_cache[key] = calculation()
    return self._calculateion_cache[key]

  def normalization_factors (self):
    return {
      'clusters mean': self.clusters_scaler.mean_,
      'clusters std': self.clusters_scaler.var_,
      'tracks mean': self.tracks_scaler.mean_,
      'tracks std': self.tracks_scaler.var_,
    }

  # input types

  def clusters_and_tracks_density_map (self, resulotion):    
    def calculate ():
      map = np.zeros((2, resulotion, resulotion), dtype=np.float32)
      for cluster in self.clusters:
        x, y = cluster.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[0, int(x * resulotion), int(y * resulotion)] += 1
      for track in self.tracks:
        x, y = track.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[1, int(x * resulotion), int(y * resulotion)] += 1
      return map

    return self.calculate_and_cache('clusters_and_tracks_density_map', calculate)
  
  def clusters_and_tracks_momentum_map (self, resulotion):
    def calculate ():
      map = np.zeros((2, resulotion, resulotion), dtype=np.float32)
      for cluster in self.clusters:
        x, y = cluster.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          momentum = cluster.momentum()
          map[0, int(x * resulotion), int(y * resulotion)] += momentum.p_t
      for track in self.tracks:
        x, y = track.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[1, int(x * resulotion), int(y * resulotion)] += track.pt
      return map

    return self.calculate_and_cache('clusters_and_tracks_momentum_map', calculate)
  
  def clusters_map (self, resulotion, channels_provider, channels_count):
    def calculate ():
      map = np.zeros((channels_count, resulotion, resulotion), dtype=np.float32)
      for cluster in self.clusters:
        x, y = cluster.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          for index, channel in enumerate(channels_provider(cluster)):
            map[index, int(x * resulotion), int(y * resulotion)] += channel
      return map

    return self.calculate_and_cache('clusters_map', calculate)

  def tracks_map (self, resulotion, channels_provider, channels_count):
    def calculate():
      map = np.zeros((channels_count, resulotion, resulotion), dtype=np.float32)
      for track in self.tracks:
        x, y = track.position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          for index, channel in enumerate(channels_provider(track)):
            map[index, int(x * resulotion), int(y * resulotion)] += channel
      return map

    return self.calculate_and_cache('tracks_map', calculate)
  
  # target types
  
  def true_position (self):
    return self.calculate_and_cache('true_position', lambda: [truth.visible_position() for truth in self.truths])
  
  def true_position_map (self, resulotion):
    def calculate ():
      map = np.zeros((resulotion, resulotion), dtype=np.float32)
      for truth in self.truths:
        x, y = truth.visible_position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[int(x * resulotion), int(y * resulotion)] += 1
      return map
    
    return self.calculate_and_cache('true_position_map', calculate)
  
  def true_momentum_map (self, resulotion):
    def calculate ():
      map = np.zeros((resulotion, resulotion), dtype=np.float32)
      for truth in self.truths:
        x, y = truth.visible_position().relative()
        if (x > 0 and x < 1 and y > 0 and y < 1):
          map[int(x * resulotion), int(y * resulotion)] += truth.visible_momentum().p_t
      return map
    
    return self.calculate_and_cache('true_momentum_map', calculate)
  
  def true_four_momentum (self):
    return self.calculate_and_cache('true_four_momentum', lambda: [truth.visible_momentum() for truth in self.truths])
  
  def total_visible_four_momentum (self):
    return self.calculate_and_cache('total_visible_momentum', lambda: sum([truth.visible_momentum() for truth in self.truths], Momentum4(0,0,0,0)))
  
  def angular_distance_between_taus (self):
    return self.calculate_and_cache('angular_distance_between_taus', lambda: self.truths[0].visible_position().distance(self.truths[1].visible_position()))
  
  def leading_pt (self):
    return self.calculate_and_cache('leading_pt', lambda: max([truth.visible_momentum().p_t for truth in self.truths]))
  
  def subleading_pt (self):
    return self.calculate_and_cache('subleading_pt', lambda: min([truth.visible_momentum().p_t for truth in self.truths]))
  
  def mass_by_channel_number (self):
    return self.calculate_and_cache('mass_by_channel_number', lambda: f'{int(20 + 10 * (self.mc_channel_number - CHANNEL_START))} GeV')
